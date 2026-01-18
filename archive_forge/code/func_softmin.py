import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
def softmin(a: TensorLikeType, dim: Optional[int]=None, _stacklevel: int=3, dtype: Optional[torch.dtype]=None) -> TensorLikeType:
    torch._check(dim is not None, lambda: 'implicit dim not supported, use dim=X')
    return torch.softmax(a=-a, dim=dim, dtype=dtype)