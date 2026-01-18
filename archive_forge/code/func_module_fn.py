import copy
import functools
import logging
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.distributed_c10d import _get_pg_default_device
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle
from torch.distributed.fsdp._fsdp_extensions import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.utils._pytree import tree_map_only
def module_fn(module, prefix, tree_level, fqn_to_param_info):
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state is None:
        return
    _lazy_init(fsdp_state, module)
    handle = _module_handle(fsdp_state, module)
    if not handle:
        return
    flat_param = handle.flat_param
    fsdp_param_info = FSDPParamInfo(fsdp_state, handle, {}, [])
    for idx, local_fqn in enumerate(flat_param._fqns):
        fqn = clean_tensor_name(prefix + local_fqn)
        if fqn in fqn_to_param_info:
            assert fqn_to_param_info[fqn].handle.flat_param is flat_param, fqn
        fqn_to_param_info[fqn] = fsdp_param_info
        fsdp_param_info.param_indices[fqn] = idx
        if flat_param._params is not None:
            fsdp_param_info.param_requires_grad.append(flat_param._params[idx].requires_grad)