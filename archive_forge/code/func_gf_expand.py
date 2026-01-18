from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_expand(F, p, K):
    """
    Expand results of :func:`~.factor` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_expand

    >>> gf_expand([([3, 2, 4], 1), ([2, 2], 2), ([3, 1], 3)], 5, ZZ)
    [4, 3, 0, 3, 0, 1, 4, 1]

    """
    if isinstance(F, tuple):
        lc, F = F
    else:
        lc = K.one
    g = [lc]
    for f, k in F:
        f = gf_pow(f, k, p, K)
        g = gf_mul(g, f, p, K)
    return g