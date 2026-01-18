import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
@classmethod
def from_arguments(cls, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: Optional[Union[torch.Tensor, AttentionBias]]=None, p: float=0.0, scale: Optional[float]=None) -> 'AttentionOpDispatch':
    """Here for backward compatibility"""
    from .dispatch import _dispatch_bw, _dispatch_fw
    inp = Inputs(query=query, key=key, value=value, attn_bias=attn_bias, p=p, scale=scale)
    return AttentionOpDispatch(op=(_dispatch_fw(inp, True), _dispatch_bw(inp)))