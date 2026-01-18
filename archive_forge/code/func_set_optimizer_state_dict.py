import contextlib
import functools
import gc
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import (
from torch.distributed.fsdp import (
from torch.distributed.fsdp._common_utils import (
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
def set_optimizer_state_dict(model: nn.Module, optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]], *, optim_state_dict: OptimizerStateType, options: Optional[StateDictOptions]=None) -> None:
    """Load the optimizers state_dict.

    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
    optimizers. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        None
    """
    with gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(model, optimizers, optim_only=True, options=options)
        _verify_state_dict({}, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)