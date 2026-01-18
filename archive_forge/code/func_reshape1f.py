import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape1f(self, array: FloatsXd, d0: int) -> Floats1d:
    return cast(Floats1d, self.reshape(array, (d0,)))