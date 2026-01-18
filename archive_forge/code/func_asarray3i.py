import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray3i(self, data: Union[Ints3d, Sequence[Sequence[Sequence[int]]]], *, dtype: Optional[DTypes]='int32') -> Ints3d:
    return cast(Ints3d, self.asarray(data, dtype=dtype))