import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reduce_last(self, X: Floats2d, lengths: Ints1d) -> Tuple[Floats2d, Ints1d]:
    if lengths.size == 0:
        return (self.alloc2f(0, X.shape[1]), lengths)
    if not self.xp.all(lengths > 0):
        raise ValueError(f'all sequence lengths must be > 0')
    lasts = lengths.cumsum() - 1
    if lasts[-1] + 1 != X.shape[0]:
        raise IndexError('lengths must sum up to the number of rows')
    return (X[lasts], lasts)