import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def dtanh(self, Y: FloatsT, *, inplace: bool=False) -> FloatsT:
    if inplace:
        Y **= 2
        Y *= -1.0
        Y += 1.0
        return Y
    else:
        return 1 - Y ** 2