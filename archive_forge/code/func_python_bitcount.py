import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def python_bitcount(n):
    """Calculate bit size of the nonnegative integer n."""
    bc = bisect(powers, n)
    if bc != 300:
        return bc
    bc = int(math.log(n, 2)) - 4
    return bc + bctable[n >> bc]