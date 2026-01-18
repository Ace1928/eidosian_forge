import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import (
import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.overrides import is_overridden
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT, _Connector, _is_using_cli
from lightning_fabric.loggers import Logger
from lightning_fabric.plugins import Precision  # avoid circular imports: # isort: split
from lightning_fabric.strategies import (
from lightning_fabric.strategies.fsdp import _has_meta_device_parameters
from lightning_fabric.strategies.launchers import _MultiProcessingLauncher, _XLALauncher
from lightning_fabric.strategies.strategy import TBroadcast, _Sharded
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from lightning_fabric.utilities.data import (
from lightning_fabric.utilities.device_dtype_mixin import _update_properties
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from lightning_fabric.utilities.registry import _load_external_callbacks
from lightning_fabric.utilities.seed import seed_everything
from lightning_fabric.utilities.types import ReduceOp
from lightning_fabric.utilities.warnings import PossibleUserWarning
from lightning_fabric.wrappers import (
def load_raw(self, path: Union[str, Path], obj: Union[nn.Module, Optimizer], strict: bool=True) -> None:
    """Load the state of a module or optimizer from a single state-dict file.

        Use this for loading a raw PyTorch model checkpoint created without Fabric.
        This is conceptually equivalent to ``obj.load_state_dict(torch.load(path))``, but is agnostic to the strategy
        being used.

        Args:
            path: A path to where the file is located
            obj: A :class:`~torch.nn.Module` or :class:`~torch.optim.Optimizer` instance.
            strict: Whether to enforce that the keys in the module's state-dict match the keys in the checkpoint.
                Does not apply to optimizers.

        """
    obj = _unwrap_objects(obj)
    self._strategy.load_checkpoint(path=path, state=obj, strict=strict)