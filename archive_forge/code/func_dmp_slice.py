from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_slice(f, m, n, u, K):
    """Take a continuous subsequence of terms of ``f`` in ``K[X]``. """
    return dmp_slice_in(f, m, n, 0, u, K)