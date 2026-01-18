import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def isqrt_small_python(x):
    """
    Correctly (floor) rounded integer square root, using
    division. Fast up to ~200 digits.
    """
    if not x:
        return x
    if x < _1_800:
        if x < _1_50:
            return int(x ** 0.5)
        r = int(x ** 0.5 * 1.00000000000001) + 1
    else:
        bc = bitcount(x)
        n = bc // 2
        r = int((x >> 2 * n - 100) ** 0.5 + 2) << n - 50
    while 1:
        y = r + x // r >> 1
        if y >= r:
            return r
        r = y