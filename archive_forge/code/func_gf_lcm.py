from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_lcm(f, g, p, K):
    """
    Compute polynomial LCM in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_lcm

    >>> gf_lcm(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    [1, 2, 0, 4]

    """
    if not f or not g:
        return []
    h = gf_quo(gf_mul(f, g, p, K), gf_gcd(f, g, p, K), p, K)
    return gf_monic(h, p, K)[1]