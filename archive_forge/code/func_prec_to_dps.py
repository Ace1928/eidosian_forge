import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def prec_to_dps(n):
    """Return number of accurate decimals that can be represented
    with a precision of n bits."""
    return max(1, int(round(int(n) / 3.3219280948873626) - 1))