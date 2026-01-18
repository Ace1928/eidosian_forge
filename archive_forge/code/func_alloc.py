import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def alloc(self, shape: Shape, *, dtype: Optional[DTypes]='float32', zeros: bool=True) -> Any:
    """Allocate an array of a certain shape."""
    if isinstance(shape, int):
        shape = (shape,)
    if zeros:
        return self.xp.zeros(shape, dtype=dtype)
    else:
        return self.xp.empty(shape, dtype=dtype)