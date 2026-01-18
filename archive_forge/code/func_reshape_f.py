import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def reshape_f(self, array: FloatsXd, shape: Shape) -> FloatsXd:
    return self.reshape(array, shape)