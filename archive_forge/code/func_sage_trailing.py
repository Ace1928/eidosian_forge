import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def sage_trailing(n):
    return MPZ(n).trailing_zero_bits()