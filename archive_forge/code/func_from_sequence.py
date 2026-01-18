import sys
from typing import Dict
@staticmethod
def from_sequence(seq):
    """Convert a sequence object into a StaticTuple instance."""
    if isinstance(seq, StaticTuple):
        return seq
    return StaticTuple(*seq)