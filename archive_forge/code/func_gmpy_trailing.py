import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def gmpy_trailing(n):
    """Count the number of trailing zero bits in abs(n) using gmpy."""
    if n:
        return MPZ(n).scan1()
    else:
        return 0