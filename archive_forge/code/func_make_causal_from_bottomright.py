import math
from dataclasses import dataclass
from typing import (
import torch
def make_causal_from_bottomright(self) -> 'BlockDiagonalCausalFromBottomRightMask':
    """Makes each block causal with a possible non-causal prefix"""
    return BlockDiagonalCausalFromBottomRightMask(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, _batch_sizes=self._batch_sizes)