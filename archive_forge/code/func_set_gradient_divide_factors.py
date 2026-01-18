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
def set_gradient_divide_factors(self, pre: float, post: float, recursive: bool) -> None:
    """Allowing user to override the pre and post divide factors.

        Args:
            pre (float): divide factor before the reduction.
            post (float): divide factor after the reduction.
            recursive (bool): recursively set it for all child FSDP instances or not.
        """
    self.assert_state(TrainingState.IDLE)
    if recursive:
        for module in self.modules():
            if isinstance(module, FullyShardedDataParallel) and module != self:
                module.set_gradient_divide_factors(pre, post, False)
    self.gradient_predivide_factor = pre
    self.gradient_postdivide_factor = post