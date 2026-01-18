from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_sqf_part(f, p, K):
    """
    Return square-free part of a ``GF(p)[x]`` polynomial.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_part

    >>> gf_sqf_part(ZZ.map([1, 1, 3, 0, 1, 0, 2, 2, 1]), 5, ZZ)
    [1, 4, 3]

    """
    _, sqf = gf_sqf_list(f, p, K)
    g = [K.one]
    for f, _ in sqf:
        g = gf_mul(g, f, p, K)
    return g