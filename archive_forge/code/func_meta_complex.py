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
@register_meta([aten.complex.default, aten.complex.out])
@out_wrapper()
def meta_complex(real, imag):
    assert real.dtype.is_floating_point
    assert imag.dtype.is_floating_point
    out_shape = _broadcast_shapes(real.shape, imag.shape)
    return real.new_empty(out_shape, dtype=corresponding_complex_dtype(real.dtype))