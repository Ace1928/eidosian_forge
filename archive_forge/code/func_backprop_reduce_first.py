import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_reduce_first(self, d_firsts: Floats2d, starts_ends: Ints1d) -> Floats2d:
    if starts_ends.size == 0:
        return self.alloc2f(0, d_firsts.shape[1], dtype=d_firsts.dtype, zeros=True)
    elif starts_ends.size == 1:
        raise ValueError(f'starts_ends must not have size 1')
    dX = self.alloc2f(int(starts_ends[-1]), d_firsts.shape[1], dtype=d_firsts.dtype, zeros=True)
    dX[starts_ends[:-1]] = d_firsts
    return dX