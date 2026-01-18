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
@register_meta([aten._scaled_dot_product_flash_attention])
def meta__scaled_dot_product_flash(query: Tensor, key: Tensor, value: Tensor, dropout_p: float=0.0, is_causal: bool=False, return_debug_mask: bool=False, scale: Optional[float]=None):
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_seqlen_batch_q = query.size(2)
    head_dim = query.size(3)
    max_seqlen_batch_k = key.size(2)
    if device_hint(query) == 'cpu':
        attention = torch.empty((batch_size, max_seqlen_batch_q, num_heads, head_dim), dtype=query.dtype, device=query.device).transpose(1, 2)
        logsumexp = torch.empty((batch_size, max_seqlen_batch_q, num_heads), dtype=torch.float, device=query.device).transpose(1, 2)
        return (attention, logsumexp, torch.empty((), dtype=torch.int32, device='meta'), torch.empty((), dtype=torch.int32, device='meta'), 0, 0, torch.empty((), dtype=torch.long, device='meta'), torch.empty((), dtype=torch.long, device='meta'), torch.empty((), dtype=query.dtype, device=query.device))
    query_t = query.transpose(1, 2)
    attention = torch.empty_like(query_t).transpose(1, 2)
    logsumexp = torch.empty((batch_size, num_heads, max_seqlen_batch_q), dtype=torch.float, device=query.device)
    if return_debug_mask:
        blocksize_c = 128 if head_dim > 64 else 256
        max_seqlen_k = math.ceil(max_seqlen_batch_q / blocksize_c)
        if max_seqlen_batch_k <= 128:
            max_seqlen_k = 128
        elif max_seqlen_batch_k <= 256:
            max_seqlen_k = 256
        debug_mask = torch.empty((batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k), dtype=query.dtype, device=query.device)
    else:
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    return (attention, logsumexp, None, None, max_seqlen_batch_q, max_seqlen_batch_k, torch.empty((), dtype=torch.long, device='meta'), torch.empty((), dtype=torch.long, device='meta'), debug_mask)