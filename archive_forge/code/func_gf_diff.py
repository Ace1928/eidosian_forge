from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_diff(f, p, K):
    """
    Differentiate polynomial in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_diff

    >>> gf_diff([3, 2, 4], 5, ZZ)
    [1, 2]

    """
    df = gf_degree(f)
    h, n = ([K.zero] * df, df)
    for coeff in f[:-1]:
        coeff *= K(n)
        coeff %= p
        if coeff:
            h[df - n] = coeff
        n -= 1
    return gf_strip(h)