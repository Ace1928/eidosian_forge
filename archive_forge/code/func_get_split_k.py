from typing import Any, List, Optional, Set, Tuple
import torch
from xformers.ops.common import get_xformers_operator, register_operator
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from xformers.ops.fmha.common import (
@classmethod
def get_split_k(cls, B: int, H: int, Mk: int) -> int:
    """Heuristic for the number of splits"""
    bh = max(B * H, 1)
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    split_k = min(split_k, 64)
    split_k = max(split_k, 1)
    return split_k