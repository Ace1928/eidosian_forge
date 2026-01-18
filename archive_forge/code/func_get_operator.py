import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
@classmethod
@functools.lru_cache
def get_operator(cls, splitk: int, *, block_m: Optional[int]=None, block_n: Optional[int]=None, num_warps: Optional[int]=None, num_stages: Optional[int]=None) -> Type[AttentionFwOpBase]:
    kwargs = {'NAME': f'triton_splitK{splitk}', 'SPLIT_K': splitk}
    if block_m is not None:
        kwargs['BLOCK_M'] = block_m
    if block_n is not None:
        kwargs['BLOCK_N'] = block_n
    if num_warps is not None:
        kwargs['NUM_WARPS'] = num_warps
    if num_stages is not None:
        kwargs['NUM_STAGES'] = num_stages
    return type(f'FwOp_S{splitk}', (cls,), kwargs)