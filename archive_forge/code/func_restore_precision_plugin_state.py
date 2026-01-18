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
def restore_precision_plugin_state(self) -> None:
    """Restore the precision plugin state from the pre-loaded checkpoint."""
    prec_plugin = self.trainer.precision_plugin
    prec_plugin.on_load_checkpoint(self._loaded_checkpoint)
    if prec_plugin.__class__.__qualname__ in self._loaded_checkpoint:
        prec_plugin.load_state_dict(self._loaded_checkpoint[prec_plugin.__class__.__qualname__])
    if 'native_amp_scaling_state' in self._loaded_checkpoint and isinstance(prec_plugin, MixedPrecision):
        prec_plugin.load_state_dict(self._loaded_checkpoint['native_amp_scaling_state'])