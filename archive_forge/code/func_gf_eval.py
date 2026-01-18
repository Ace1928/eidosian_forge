from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_eval(f, a, p, K):
    """
    Evaluate ``f(a)`` in ``GF(p)`` using Horner scheme.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_eval

    >>> gf_eval([3, 2, 4], 2, 5, ZZ)
    0

    """
    result = K.zero
    for c in f:
        result *= a
        result += c
        result %= p
    return result