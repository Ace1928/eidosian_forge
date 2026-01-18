import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def alloc_i(self, shape: Shape, *, dtype: Optional[DTypesInt]='int32', zeros: bool=True) -> IntsXd:
    return cast(IntsXd, self.alloc(shape, dtype=dtype, zeros=zeros))