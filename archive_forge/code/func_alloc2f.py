import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def alloc2f(self, d0: int, d1: int, *, dtype: Optional[DTypesFloat]='float32', zeros: bool=True) -> Floats2d:
    return cast(Floats2d, self.alloc((d0, d1), dtype=dtype, zeros=zeros))