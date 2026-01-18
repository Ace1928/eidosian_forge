import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def python_trailing(n):
    """Count the number of trailing zero bits in abs(n)."""
    if not n:
        return 0
    low_byte = n & 255
    if low_byte:
        return small_trailing[low_byte]
    t = 8
    n >>= 8
    while not n & 255:
        n >>= 8
        t += 8
    return t + small_trailing[n & 255]