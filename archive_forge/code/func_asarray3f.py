import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def asarray3f(self, data: Union[Floats3d, Sequence[Sequence[Sequence[float]]]], *, dtype: Optional[DTypes]='float32') -> Floats3d:
    return cast(Floats3d, self.asarray(data, dtype=dtype))