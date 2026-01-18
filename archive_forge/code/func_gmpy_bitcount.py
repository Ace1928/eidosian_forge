import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def gmpy_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    if n:
        return MPZ(n).numdigits(2)
    else:
        return 0