import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import os
import time
import traceback
import typing
from typing import (
import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairscale.internal.containers import apply_to_tensors
from fairscale.internal.parallel import (
from fairscale.internal.params import calc_grad_norm, recursive_copy_to_device
from fairscale.internal.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.internal.state_dict import replace_by_prefix_
from fairscale.nn.misc import FlattenParamsWrapper, _enable_pre_load_state_dict_hook
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from . import fsdp_optim_utils as ou
def get_fsdp_instances(mod: nn.Module, skip_empty: bool=False) -> List[FullyShardedDataParallel]:
    """Return, a list, if any, of the module/submodule is wrapped by FSDP within another module.

    Args:
        mod (nn.Module):
            A nn.Module module to be scanned.
        skip_empty (bool):
            If True, skip wrappers without any parameters.
            Default: False
    """
    ret: List[FullyShardedDataParallel] = []
    for m in mod.modules():
        if isinstance(m, FullyShardedDataParallel):
            ret.append(m)
    if skip_empty:
        ret = list(filter(lambda x: len(cast(FullyShardedDataParallel, x).non_shared_params()) > 0, ret))
    return ret