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
@register_meta([aten._fft_c2c.default, aten._fft_c2c.out])
@out_wrapper()
def meta_fft_c2c(self, dim, normalization, forward):
    assert self.dtype.is_complex
    out_sizes = self.shape
    output = self.new_empty(out_sizes)
    if not dim:
        return output
    sorted_dims = dim[:]
    self_strides = self.stride()
    sorted_dims.sort(key=lambda x: self_strides[x], reverse=True)
    output = _exec_fft(output, self, out_sizes, sorted_dims, forward)
    return output