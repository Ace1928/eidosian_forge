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
@register_meta([aten._flash_attention_backward])
def meta__flash_attention_backward(grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, out: Tensor, logsumexp: Tensor, cum_seq_q: Tensor, cum_seq_k: Tensor, max_q: int, max_k: int, dropout_p: float, is_causal: bool, philox_seed: Tensor, philox_offset: Tensor, scale: Optional[float]=None):
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    return (grad_query, grad_key, grad_value)