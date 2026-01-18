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
def is_sharded(self, tensor: Tensor) -> bool:
    """
        Returns if ``tensor`` is *currently* sharded. For ``NO_SHARD``, we
        choose to have this always return ``False`` for clarity.
        """
    if not hasattr(self.flat_param, '_sharded_size') or not self.uses_sharded_strategy:
        return False
    sharded_size = self.flat_param._sharded_size
    return tensor.size() == sharded_size