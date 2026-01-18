import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray1f(self, data: Union[Floats1d, Sequence[float]], *, dtype: Optional[DTypes]='float32') -> Floats1d:
    return cast(Floats1d, self.asarray(data, dtype=dtype))