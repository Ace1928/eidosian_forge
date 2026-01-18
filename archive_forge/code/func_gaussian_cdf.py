import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def gaussian_cdf(ops: Ops, X: FloatsXdT) -> FloatsXdT:
    """Gaussian CDF for distribution with mean 0 and stdev 1."""
    return 0.5 * (1.0 + ops.erf(INV_SQRT2 * X))