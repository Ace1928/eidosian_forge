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
def resume_start(self, checkpoint_path: Optional[_PATH]=None) -> None:
    """Attempts to pre-load the checkpoint file to memory, with the source path determined in this priority:

        1. from HPC weights if `checkpoint_path` is ``None`` and on SLURM or passed keyword `"hpc"`.
        2. from fault-tolerant auto-saved checkpoint if found
        3. from `checkpoint_path` file if provided
        4. don't restore

        """
    self._ckpt_path = checkpoint_path
    if not checkpoint_path:
        log.debug('`checkpoint_path` not specified. Skipping checkpoint loading.')
        return
    rank_zero_info(f'Restoring states from the checkpoint path at {checkpoint_path}')
    with pl_legacy_patch():
        loaded_checkpoint = self.trainer.strategy.load_checkpoint(checkpoint_path)
    self._loaded_checkpoint = _pl_migrate_checkpoint(loaded_checkpoint, checkpoint_path)