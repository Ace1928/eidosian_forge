import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
def reshard_grad(self):
    if self._use_orig_params:
        self._use_sharded_grad_views()
    if not self.uses_sharded_strategy:
        return
    self.flat_param.grad = self.flat_param._saved_grad_shard
    delattr(self.flat_param, '_saved_grad_shard')