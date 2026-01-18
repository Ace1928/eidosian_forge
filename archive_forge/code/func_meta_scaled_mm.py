import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta([aten._scaled_mm.default])
def meta_scaled_mm(self: torch.Tensor, mat2: torch.Tensor, bias: Optional[torch.Tensor]=None, out_dtype: Optional[torch.dtype]=None, scale_a: Optional[torch.Tensor]=None, scale_b: Optional[torch.Tensor]=None, scale_result: Optional[torch.Tensor]=None, use_fast_accum: bool=False):

    def is_row_major(stride):
        return stride[0] > stride[1] and stride[1] == 1

    def is_col_major(shape, stride):
        return stride[0] == 1 and stride[1] == shape[0]

    def is_fp8_type(dtype):
        return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    torch._check(self.dim() == 2 and mat2.dim() == 2, lambda: f'Inputs must be 2D but got self.dim()={self.dim()} and mat2.dim()={mat2.dim()}')
    torch._check(is_row_major(self.stride()), lambda: 'self must be row_major')
    torch._check(is_col_major(mat2.shape, mat2.stride()), lambda: 'mat2 must be col_major')
    torch._check(self.size(1) % 16 == 0, lambda: f'Expected self.size(0) to be divisible by 16, but got self.size(1)={self.size(1)}')
    torch._check(mat2.size(0) % 16 == 0 and mat2.size(1) % 16 == 0, lambda: f'Expected both dimensions of mat2 to be divisble by 16 but got {mat2.shape}')
    torch._check(is_fp8_type(self.dtype) and is_fp8_type(mat2.dtype), lambda: f'Expected both inputs to be fp8 types but got self.dtype={self.dtype} and mat2.dtype={mat2.dtype}')
    _out_dtype = out_dtype if out_dtype is not None else self.dtype
    return (torch.empty(self.size(0), mat2.size(1), dtype=_out_dtype, device=self.device), torch.empty((), dtype=torch.float32, device=self.device))