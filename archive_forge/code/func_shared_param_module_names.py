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
def shared_param_module_names(self) -> Iterator[Tuple[str, str]]:
    for param_name, _, module_name in [ParamInfo(param_name, module, module_name) for param_name, module, module_name, _, _, _ in self.flat_param._shared_param_infos]:
        yield (param_name, module_name)