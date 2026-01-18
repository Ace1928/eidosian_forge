import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray4f(self, data: Union[Floats4d, Sequence[Sequence[Sequence[Sequence[float]]]]], *, dtype: Optional[DTypes]='float32') -> Floats4d:
    return cast(Floats4d, self.asarray(data, dtype=dtype))