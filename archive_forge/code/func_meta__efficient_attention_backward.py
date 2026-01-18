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
@register_meta([aten._efficient_attention_backward])
def meta__efficient_attention_backward(grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, bias: Optional[Tensor], cu_seqlens_q: Optional[Tensor], cu_seqlens_k: Optional[Tensor], max_seqlen_q: int, max_seqlen_k: int, logsumexp: Tensor, dropout_p: float, philox_seed: Tensor, philox_offset: Tensor, custom_mask_type: int, bias_requires_grad: bool, scale: Optional[float]=None, num_splits_key: Optional[int]=None):
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    if bias is not None:
        lastDim = bias.size(-1)
        lastDimAligned = lastDim if lastDim % 16 == 0 else lastDim + 16 - lastDim % 16
        new_sizes = list(bias.size())
        new_sizes[-1] = lastDimAligned
        grad_bias = torch.empty(new_sizes, dtype=bias.dtype, device=bias.device)
        grad_bias = grad_bias[..., :lastDim]
    else:
        grad_bias = torch.empty((), device=query.device)
    return (grad_query, grad_key, grad_value, grad_bias)