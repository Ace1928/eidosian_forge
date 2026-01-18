import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def sqrtrem_python(x):
    """Correctly rounded integer (floor) square root with remainder."""
    if x < _1_600:
        y = isqrt_small_python(x)
        return (y, x - y * y)
    y = isqrt_fast_python(x) + 1
    rem = x - y * y
    while rem < 0:
        y -= 1
        rem += 1 + 2 * y
    else:
        if rem:
            while rem > 2 * (1 + y):
                y += 1
                rem -= 1 + 2 * y
    return (y, rem)