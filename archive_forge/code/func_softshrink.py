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
@aten.softshrink.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.softshrink)
@out_wrapper()
def softshrink(a: TensorLikeType, lambd: float=0.5):
    torch._check(lambd >= 0, lambda: f'lambda must be greater or equal to 0, but found to be {lambd}')
    return torch.where(torch.abs(a) > lambd, a - torch.sign(a) * lambd, 0)