import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape4f(self, array: FloatsXd, d0: int, d1: int, d2: int, d3: int) -> Floats4d:
    return cast(Floats4d, self.reshape(array, (d0, d1, d2, d3)))