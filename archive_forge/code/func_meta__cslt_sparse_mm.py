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
@register_meta(aten._cslt_sparse_mm)
def meta__cslt_sparse_mm(compressed_A: torch.Tensor, dense_B: torch.Tensor, bias: Optional[Tensor]=None, alpha: Optional[Tensor]=None, out_dtype: Optional[torch.dtype]=None, transpose_result: bool=False):
    assert dense_B.dtype in {torch.float16, torch.bfloat16, torch.int8}, '_cslt_sparse_mm only supports fp16, bf16, and int8'
    assert compressed_A.dtype == dense_B.dtype, 'inputs must have the same dtype'
    assert len(dense_B.shape) == 2, '_cslt_sparse_mm only supports 2d inputs'
    is_int8_input_type = compressed_A.dtype == torch.int8
    compression_factor = 10 if is_int8_input_type else 9
    k = dense_B.size(0)
    n = dense_B.size(1)
    m = compressed_A.numel() * 16 // (compression_factor * k)
    if bias is not None:
        assert m == bias.size(0)
    if out_dtype is not None:
        assert is_int8_input_type and out_dtype == torch.float16, 'out_dtype is only supported for i8i8->fp16 matmul'
    output_shape = (n, m) if transpose_result else (m, n)
    result = dense_B.new_empty(output_shape, dtype=out_dtype)
    return result