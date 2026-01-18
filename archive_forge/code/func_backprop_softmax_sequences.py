import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_softmax_sequences(self, dY: Floats2d, Y: Floats2d, lengths: Ints1d) -> Floats2d:
    dX = Y * dY
    sum_dX = self.backprop_reduce_sum(self.reduce_sum(dX, lengths), lengths)
    dX -= Y * sum_dX
    return dX