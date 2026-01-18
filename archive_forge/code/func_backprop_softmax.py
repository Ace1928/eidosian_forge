import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_softmax(self, Y: FloatsT, dY: FloatsT, *, axis: int=-1, temperature: float=1.0) -> FloatsT:
    if temperature != 1.0:
        dY = dY / temperature
    dX = Y * dY
    dX -= Y * dX.sum(axis=axis, keepdims=True)
    return dX