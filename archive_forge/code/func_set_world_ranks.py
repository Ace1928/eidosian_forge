import logging
import shutil
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Mapping, Optional, Set, Type, Union
import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.fsdp import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _lazy_load, _materialize_tensors
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, ReduceOp
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.plugins.precision.fsdp import FSDPPrecision
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn
def set_world_ranks(self) -> None:
    if self.cluster_environment is not None:
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
    rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank