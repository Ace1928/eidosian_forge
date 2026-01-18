import logging
import os
import re
from typing import Any, Dict, Optional
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.precision import MixedPrecision
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def restore_optimizers_and_schedulers(self) -> None:
    """Restores the optimizers and learning rate scheduler states from the pre-loaded checkpoint."""
    if not self._loaded_checkpoint:
        return
    if self.trainer.strategy.lightning_restore_optimizer:
        if 'optimizer_states' not in self._loaded_checkpoint:
            raise KeyError('Trying to restore optimizer state but checkpoint contains only the model. This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`.')
        self.restore_optimizers()
    if 'lr_schedulers' not in self._loaded_checkpoint:
        raise KeyError('Trying to restore learning rate scheduler state but checkpoint contains only the model. This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`.')
    self.restore_lr_schedulers()