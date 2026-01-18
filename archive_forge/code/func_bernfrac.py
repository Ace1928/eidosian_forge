import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def bernfrac(n):
    """
    Returns a tuple of integers `(p, q)` such that `p/q = B_n` exactly,
    where `B_n` denotes the `n`-th Bernoulli number. The fraction is
    always reduced to lowest terms. Note that for `n > 1` and `n` odd,
    `B_n = 0`, and `(0, 1)` is returned.

    **Examples**

    The first few Bernoulli numbers are exactly::

        >>> from mpmath import *
        >>> for n in range(15):
        ...     p, q = bernfrac(n)
        ...     print("%s %s/%s" % (n, p, q))
        ...
        0 1/1
        1 -1/2
        2 1/6
        3 0/1
        4 -1/30
        5 0/1
        6 1/42
        7 0/1
        8 -1/30
        9 0/1
        10 5/66
        11 0/1
        12 -691/2730
        13 0/1
        14 7/6

    This function works for arbitrarily large `n`::

        >>> p, q = bernfrac(10**4)
        >>> print(q)
        2338224387510
        >>> print(len(str(p)))
        27692
        >>> mp.dps = 15
        >>> print(mpf(p) / q)
        -9.04942396360948e+27677
        >>> print(bernoulli(10**4))
        -9.04942396360948e+27677

    .. note ::

        :func:`~mpmath.bernoulli` computes a floating-point approximation
        directly, without computing the exact fraction first.
        This is much faster for large `n`.

    **Algorithm**

    :func:`~mpmath.bernfrac` works by computing the value of `B_n` numerically
    and then using the von Staudt-Clausen theorem [1] to reconstruct
    the exact fraction. For large `n`, this is significantly faster than
    computing `B_1, B_2, \\ldots, B_2` recursively with exact arithmetic.
    The implementation has been tested for `n = 10^m` up to `m = 6`.

    In practice, :func:`~mpmath.bernfrac` appears to be about three times
    slower than the specialized program calcbn.exe [2]

    **References**

    1. MathWorld, von Staudt-Clausen Theorem:
       http://mathworld.wolfram.com/vonStaudt-ClausenTheorem.html

    2. The Bernoulli Number Page:
       http://www.bernoulli.org/

    """
    n = int(n)
    if n < 3:
        return [(1, 1), (-1, 2), (1, 6)][n]
    if n & 1:
        return (0, 1)
    q = 1
    for k in list_primes(n + 1):
        if not n % (k - 1):
            q *= k
    prec = bernoulli_size(n) + int(math.log(q, 2)) + 20
    b = mpf_bernoulli(n, prec)
    p = mpf_mul(b, from_int(q))
    pint = to_int(p, round_nearest)
    return (pint, q)