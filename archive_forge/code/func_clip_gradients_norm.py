import shutil
from contextlib import ExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
from lightning_fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _materialize_tensors, _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, _Stateful
@override
def clip_gradients_norm(self, module: Module, optimizer: Optimizer, max_norm: Union[float, int], norm_type: Union[float, int]=2.0, error_if_nonfinite: bool=True) -> Tensor:
    """Clip gradients by norm."""
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
    if not isinstance(module, FullyShardedDataParallel):
        raise TypeError(f'Gradient clipping with FSDP is only possible if the module passed to `{self.__class__.__name__}.clip_gradients_norm` is wrapped in `FullyShardedDataParallel`. Got: {module.__class__.__name__}.')
    self.precision.unscale_gradients(optimizer)
    return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)