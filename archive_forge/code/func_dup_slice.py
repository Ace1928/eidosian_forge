from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_slice(f, m, n, K):
    """Take a continuous subsequence of terms of ``f`` in ``K[x]``. """
    k = len(f)
    if k >= m:
        M = k - m
    else:
        M = 0
    if k >= n:
        N = k - n
    else:
        N = 0
    f = f[N:M]
    if not f:
        return []
    else:
        return f + [K.zero] * m