import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape4(self, array: ArrayXd, d0: int, d1: int, d2: int, d3: int) -> Array4d:
    return cast(Array4d, self.reshape(array, (d0, d1, d2, d3)))