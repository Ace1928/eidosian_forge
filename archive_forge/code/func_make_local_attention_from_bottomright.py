import math
from dataclasses import dataclass
from typing import (
import torch
def make_local_attention_from_bottomright(self, window_size: int) -> 'BlockDiagonalCausalLocalAttentionFromBottomRightMask':
    """Experimental: Makes each block causal with local attention, start from bottom right"""
    return BlockDiagonalCausalLocalAttentionFromBottomRightMask(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, _batch_sizes=self._batch_sizes, _window_size=window_size)