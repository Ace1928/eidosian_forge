from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_sub(f, g, p, K):
    """
    Subtract polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub

    >>> gf_sub([3, 2, 4], [2, 2, 2], 5, ZZ)
    [1, 0, 2]

    """
    if not g:
        return f
    if not f:
        return gf_neg(g, p, K)
    df = gf_degree(f)
    dg = gf_degree(g)
    if df == dg:
        return gf_strip([(a - b) % p for a, b in zip(f, g)])
    else:
        k = abs(df - dg)
        if df > dg:
            h, f = (f[:k], f[k:])
        else:
            h, g = (gf_neg(g[:k], p, K), g[k:])
        return h + [(a - b) % p for a, b in zip(f, g)]