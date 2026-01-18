import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_relu_k(self, dY: FloatsXdT, X: FloatsXdT, n: float=6.0, inplace: bool=False) -> FloatsXdT:
    return self.backprop_clipped_linear(dY, X, max_val=n, inplace=inplace)