from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def memory_efficient_attention_forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: Optional[Union[torch.Tensor, AttentionBias]]=None, p: float=0.0, scale: Optional[float]=None, *, op: Optional[Type[AttentionFwOpBase]]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    """
    Calculates the forward pass of :attr:`xformers.ops.memory_efficient_attention`.
    """
    return _memory_efficient_attention_forward(Inputs(query=query, key=key, value=value, p=p, attn_bias=attn_bias, scale=scale, output_dtype=output_dtype), op=op)