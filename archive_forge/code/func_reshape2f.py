import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape2f(self, array: FloatsXd, d0: int, d1: int) -> Floats2d:
    return cast(Floats2d, self.reshape(array, (d0, d1)))