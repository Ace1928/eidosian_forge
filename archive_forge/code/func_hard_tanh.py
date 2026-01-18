import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def hard_tanh(self, X: FloatsXdT, inplace: bool=False) -> FloatsXdT:
    return self.clipped_linear(X, min_val=-1.0, max_val=1.0, inplace=inplace)