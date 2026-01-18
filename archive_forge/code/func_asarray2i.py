import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray2i(self, data: Union[Ints2d, Sequence[Sequence[int]]], *, dtype: Optional[DTypes]='int32') -> Ints2d:
    return cast(Ints2d, self.asarray(data, dtype=dtype))