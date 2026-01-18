import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def numeral_gmpy(n, base=10, size=0, digits=stddigits):
    """Represent the integer n as a string of digits in the given base.
    Recursive division is used to make this function about 3x faster
    than Python's str() for converting integers to decimal strings.

    The 'size' parameters specifies the number of digits in n; this
    number is only used to determine splitting points and need not be
    exact."""
    if n < 0:
        return '-' + numeral(-n, base, size, digits)
    if size < 1500000:
        return gmpy.digits(n, base)
    half = size // 2 + (size & 1)
    A, B = divmod(n, MPZ(base) ** half)
    ad = numeral(A, base, half, digits)
    bd = numeral(B, base, half, digits).rjust(half, '0')
    return ad + bd