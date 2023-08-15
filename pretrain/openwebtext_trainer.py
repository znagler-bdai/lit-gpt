import math
import sys
import time
from pathlib import Path
from typing import Optional, Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import FSDPStrategy, XLAStrategy, DeepSpeedStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from lightning.pytorch.profilers import PyTorchProfiler
from deepspeed import ops

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.speed_monitor import measure_flops, estimate_flops, SpeedMonitorCallback
from lit_gpt.utils import step_csv_logger, chunked_cross_entropy

name = "openwebtext"
out_dir = Path("out") / name
data_dir = Path("data") / name
save_interval = 1000
log_interval = 1


# Hyperparameters
train_bin_path = "/home/jpatel_theaiinstitute_com/dev/nanoGPT/data/openwebtext/train.bin"
val_bin_path = "/home/jpatel_theaiinstitute_com/dev/nanoGPT/data/openwebtext/val.bin"

max_steps = 1_000
gradient_accumulation_steps = 1
batch_size = 4

ds_dict = {
    "stage": 3,
    "offload_optimizer": True,
    "offload_parameters": True,
}

ds_offload_summary = "none"
if ds_dict["offload_optimizer"]:
    ds_offload_summary = "optimizer"
if ds_dict["offload_parameters"]:
    ds_offload_summary = "parameters"
if ds_dict["offload_optimizer"] and ds_dict["offload_parameters"]:
    ds_offload_summary = "both"

max_iters = max_steps * gradient_accumulation_steps
train_params = {
    "eval_batches": 10,
    "batch_size": batch_size,
    "max_steps": max_steps,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "max_iters": max_iters,
    "total_training_examples": batch_size * max_steps,  # only informational, not used
    "eval_interval": 100,
    "lightning_strategy": DeepSpeedStrategy(**ds_dict),
    # "lightning_strategy": "ddp",
    "devices": 8,
    "precision": "16-mixed",
}


learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

lr_params = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

# wandb stuff
wandb_project = "pythia-1b"  # Project
wandb_name = f"ds_offload/stage:{ds_offload_summary}/{ds_dict['stage']}"  # this will be the wandb run name
wandb_experiment_name = "deepspeed_exp2"  # currently using wandb "tags" for this


class LightningGPTModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.module: Optional[torch.nn.Module] = None
        self.measured_flops: Optional[int] = None

        self.save_hyperparameters()

    def configure_model(self) -> None:
        self.module = GPT(self.config)
        self.module.apply(self.module._init_weights)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return ops.adam.DeepSpeedCPUAdam(
            self.module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )

        # return torch.optim.AdamW(
        #     self.module.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        #     betas=(beta1, beta2),
        #     foreach=False,
        # )

    def print_stats(self):
        self.print("*" * 100)
        self.print(
            f"Running {max_steps} steps of {gradient_accumulation_steps} iters each, totaling \
            {max_iters} batches.  With batch size {batch_size}.  this covers \
            {train_params['total_training_examples']} total training examples"
        )
        self.print("*" * 100)
        trainer = self.trainer
        with torch.device("meta"):
            meta_model = GPT(self.module.config)
            # estimated is too much of an optimistic estimate, left just for reference
            estimated_flops = estimate_flops(meta_model) * batch_size
            self.print(f"Estimated TFLOPs: {estimated_flops * trainer.world_size / 1e12:.2f}")
            x = torch.randint(0, 1, (batch_size, meta_model.config.block_size))
            self.measured_flops = measure_flops(meta_model, x)
            self.print(f"Measured TFLOPs: {self.measured_flops * trainer.world_size / 1e12:.2f}")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not decay_lr:
            return
        # determine and set the learning rate for this iteration
        lr = get_lr(self.trainer.fit_loop.total_batch_idx)
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_ids, targets = batch
        if batch_idx == 0:
            self.print("training input_ids.shape", input_ids.shape, input_ids)
            self.print("training targets.shape", targets.shape, targets)
            self.print_stats()

        logits = self.module(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        # unsure about this custom cross_entropy, would prefer to use standard
        # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # tracking torch's memory allocated on one GPU for comparison
        self.log("memory cuda:0", torch.cuda.memory_allocated(torch.device("cuda:0")))

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, targets = batch
        logits = self.module(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        # unsure about this custom cross_entropy, would prefer to use standard
        # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


def main(
    model_name="pythia-70m",
) -> None:
    logger = WandbLogger(project=wandb_project, name=wandb_name, tags=[wandb_experiment_name])

    trainer = L.Trainer(
        max_steps=max_steps,
        profiler="simple",
        devices=train_params["devices"],
        accelerator="auto",
        strategy=train_params["lightning_strategy"],
        precision=train_params["precision"],
        logger=logger,
        # callbacks=[DeviceStatsMonitor()],
        max_epochs=1,
        limit_val_batches=train_params["eval_batches"],
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_interval,
        val_check_interval=train_params["eval_interval"],
        enable_model_summary=True,
    )

    L.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    trainer.print("Train params", train_params)
    trainer.print("Learning Rate params", lr_params)
    trainer.print(f"{trainer.accelerator=} {trainer.strategy=}")

    if trainer.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.experiment.config.update(train_params)  # Makes params show up in the information tab in Wandb

    config = Config.from_name(model_name)
    trainer.print(f"Loading model with {config.__dict__}")
    t0 = time.time()
    model = LightningGPTModule(config)
    trainer.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.")

    train_data = Dataset(train_bin_path, config.block_size)
    val_data = Dataset(val_bin_path, config.block_size)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=2)

    trainer.print("Calling trainer.fit...")
    t0 = time.time()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.time()-t0):.2f}s")


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, block_size: int):
        super().__init__()
        self.data_file = data_file
        self.block_size = block_size

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.block_size, (1,)).item()
            x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64))
            yield x, y


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
