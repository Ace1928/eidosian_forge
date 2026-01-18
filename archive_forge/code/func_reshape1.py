import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape1(self, array: ArrayXd, d0: int) -> Array1d:
    return cast(Array1d, self.reshape(array, (d0,)))