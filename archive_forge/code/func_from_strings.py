from dataclasses import dataclass
from typing import List
from .align import get_alignments
from .alignment_array import AlignmentArray
@classmethod
def from_strings(cls, A: List[str], B: List[str]) -> 'Alignment':
    x2y, y2x = get_alignments(A, B)
    return Alignment.from_indices(x2y=x2y, y2x=y2x)