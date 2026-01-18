import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def ifac(n, memo={0: 1, 1: 1}):
    """Return n factorial (for integers n >= 0 only)."""
    f = memo.get(n)
    if f:
        return f
    k = len(memo)
    p = memo[k - 1]
    MAX = MAX_FACTORIAL_CACHE
    while k <= n:
        p *= k
        if k <= MAX:
            memo[k] = p
        k += 1
    return p