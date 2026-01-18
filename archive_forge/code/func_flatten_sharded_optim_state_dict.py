import contextlib
import copy
import functools
import math
import traceback
import warnings
from contextlib import contextmanager
from enum import auto, Enum
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParameter
from ._optim_utils import (
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
from .wrap import CustomPolicy, ModuleWrapPolicy
@staticmethod
def flatten_sharded_optim_state_dict(sharded_optim_state_dict: Dict[str, Any], model: torch.nn.Module, optim: torch.optim.Optimizer) -> Dict[str, Any]:
    """Flatten a sharded optimizer state-dict.

        The API is similar to :meth:`shard_full_optim_state_dict`. The only
        difference is that the input ``sharded_optim_state_dict`` should be
        returned from :meth:`sharded_optim_state_dict`. Therefore, there will
        be all-gather calls on each rank to gather ``ShardedTensor`` s.

        Args:
            sharded_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                sharded optimizer state.
            model (torch.nn.Module):
                Refer to :meth:`shard_full_optim_state_dict`.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.

        Returns:
            Refer to :meth:`shard_full_optim_state_dict`.
        """
    FullyShardedDataParallel._warn_legacy_optim_state_dict('flatten_sharded_optim_state_dict', 'optim_state_dict_to_load')
    return FullyShardedDataParallel._optim_state_dict_to_load_impl(optim_state_dict=sharded_optim_state_dict, model=model, optim_input=None, optim=optim, full_state_dict=False, is_named_optimizer=False)