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
@register_meta([aten._efficient_attention_forward])
def meta__efficient_attention_forward(query: Tensor, key: Tensor, value: Tensor, bias: Optional[Tensor], cu_seqlens_q: Optional[Tensor], cu_seqlens_k: Optional[Tensor], max_seqlen_q: Optional[int], dropout_p: float, custom_mask_type: int, compute_log_sumexp: bool=False, scale: Optional[float]=None, causal_diagonal: Optional[Tensor]=None, seqlen_k: Optional[Tensor]=None):
    B = query.size(0)
    M = query.size(1)
    N = key.size(1)
    num_heads = query.size(-2)
    K = query.size(-1)
    Kv = value.size(-1)
    res = torch.empty(B, M, num_heads, Kv, dtype=query.dtype, device=query.device)
    logsumexp_dim = math.ceil(M / 32) * 32 if compute_log_sumexp else 0
    logsum_exp = torch.empty((B, num_heads, logsumexp_dim), dtype=torch.float, device=query.device)
    seed = torch.empty((), dtype=torch.long, device='meta')
    offset = torch.empty((), dtype=torch.long, device='meta')
    return (res, logsum_exp, seed, offset, M, N)