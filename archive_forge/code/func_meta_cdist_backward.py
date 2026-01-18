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
@register_meta(aten._cdist_backward)
@out_wrapper()
def meta_cdist_backward(grad, x1, x2, p, cdist):
    c1 = x1.shape[-1]
    r1 = x1.shape[-2]
    r2 = x2.shape[-2]
    batch_tensor1 = x1.shape[:-2]
    batch_tensor2 = x2.shape[:-2]
    expand_batch_portion = list(torch.broadcast_shapes(batch_tensor1, batch_tensor2))
    tensor1_expand_size = expand_batch_portion.copy()
    tensor1_expand_size.extend([r1, c1])
    batch_product = math.prod(expand_batch_portion)
    if r1 == 0 or r2 == 0 or c1 == 0 or (batch_product == 0):
        return torch.zeros_like(x1)
    if tensor1_expand_size != list(x1.shape):
        x1 = x1.expand(tensor1_expand_size)
    return torch.empty_like(x1, memory_format=torch.contiguous_format)