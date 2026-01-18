import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def cblas(self) -> CBlas:
    """Return C BLAS function table."""
    err = f'{type(self).__name__} does not provide C BLAS functions'
    raise NotImplementedError(err)