import math
from bisect import bisect
from .backend import xrange
from .backend import BACKEND, gmpy, sage, sage_utils, MPZ, MPZ_ONE, MPZ_ZERO
def moebius(n):
    """
    Evaluates the Moebius function which is `mu(n) = (-1)^k` if `n`
    is a product of `k` distinct primes and `mu(n) = 0` otherwise.

    TODO: speed up using factorization
    """
    n = abs(int(n))
    if n < 2:
        return n
    factors = []
    for p in xrange(2, n + 1):
        if not n % p:
            if not n % p ** 2:
                return 0
            if not sum((p % f for f in factors)):
                factors.append(p)
    return (-1) ** len(factors)