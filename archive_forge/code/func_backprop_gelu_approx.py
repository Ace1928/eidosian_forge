import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_gelu_approx(self, dY: FloatsXdT, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
    dX = cast(FloatsXdT, self.alloc_f(X.shape))
    Xp3 = self.xp.power(X, 3)
    tmp = 0.5 * self.xp.tanh(0.0356774 * Xp3 + 0.797885 * X)
    tmp += (0.0535161 * Xp3 + 0.398942 * X) * self.sechsq(0.0356774 * Xp3 + 0.797885 * X)
    tmp += 0.5
    dX += tmp
    if inplace:
        dY *= dX
        return dY
    return dY * dX