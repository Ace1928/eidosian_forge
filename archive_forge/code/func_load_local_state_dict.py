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
def load_local_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], 'OrderedDict[str, torch.Tensor]'], strict: bool=True) -> NamedTuple:
    """Load a local (sharded) state_dict."""
    self.assert_state([TrainingState.IDLE, TrainingState.FORWARD, TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])
    with contextlib.ExitStack() as stack:
        for module in get_fsdp_instances(self):
            stack.enter_context(module._no_return_full_state_dict())
        output = self._load_state_dict(state_dict, strict)
    for module in get_fsdp_instances(self):
        module._free_full_params()
    return output