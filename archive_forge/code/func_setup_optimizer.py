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
def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
    """Set up an optimizer for a model wrapped with FSDP.

        This setup method doesn't modify the optimizer or wrap the optimizer. The only thing it currently does is verify
        that the optimizer was created after the model was wrapped with :meth:`setup_module` with a reference to the
        flattened parameters.

        """
    if self._fsdp_kwargs.get('use_orig_params'):
        return super().setup_optimizer(optimizer)
    if not _optimizer_has_flat_params(optimizer):
        raise ValueError('The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the optimizer after setting up the model.')
    return optimizer