import logging
import os
import shutil
import tempfile
from typing import Any, Dict
import torch
from packaging.version import Version
import ray
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint
from ray.util import PublicAPI
def on_train_epoch_end(self, trainer, pl_module) -> None:
    tmpdir = os.path.join(self.tmpdir_prefix, str(trainer.current_epoch))
    os.makedirs(tmpdir, exist_ok=True)
    metrics = trainer.callback_metrics
    metrics = {k: v.item() for k, v in metrics.items()}
    metrics['epoch'] = trainer.current_epoch
    metrics['step'] = trainer.global_step
    ckpt_path = os.path.join(tmpdir, self.CHECKPOINT_NAME)
    trainer.save_checkpoint(ckpt_path, weights_only=False)
    checkpoint = Checkpoint.from_directory(tmpdir)
    train.report(metrics=metrics, checkpoint=checkpoint)
    torch.distributed.barrier()
    if self.local_rank == 0:
        shutil.rmtree(tmpdir)