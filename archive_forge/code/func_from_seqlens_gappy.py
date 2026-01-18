import math
from dataclasses import dataclass
from typing import (
import torch
@classmethod
def from_seqlens_gappy(cls, seqstarts: Sequence[int], seqlens: Sequence[int]) -> '_GappySeqInfo':
    assert not isinstance(seqlens, torch.Tensor)
    seqstart_py = list(seqstarts)
    if len(seqlens) == 0:
        raise ValueError('No elements')
    if len(seqstarts) - len(seqlens) != 1:
        raise ValueError(f'len(seqstarts) {seqstarts} should be 1 + len(seqlens) {seqlens}')
    seqlen = torch.tensor(seqlens, dtype=torch.int32)
    return cls(seqlen=seqlen, seqlen_py=seqlens, max_seqlen=max(seqlens), min_seqlen=min(seqlens), seqstart=torch.tensor(seqstart_py, dtype=torch.int32), seqstart_py=seqstart_py)