import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_reduce_last(self, d_lasts: Floats2d, lasts: Ints1d) -> Floats2d:
    if lasts.size == 0:
        return self.alloc2f(0, d_lasts.shape[1], dtype=d_lasts.dtype, zeros=True)
    dX = self.alloc2f(int(lasts[-1]) + 1, d_lasts.shape[1], dtype=d_lasts.dtype, zeros=True)
    dX[lasts] = d_lasts
    return dX