from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_compose(f, g, p, K):
    """
    Compute polynomial composition ``f(g)`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_compose

    >>> gf_compose([3, 2, 4], [2, 2, 2], 5, ZZ)
    [2, 4, 0, 3, 0]

    """
    if len(g) <= 1:
        return gf_strip([gf_eval(f, gf_LC(g, K), p, K)])
    if not f:
        return []
    h = [f[0]]
    for c in f[1:]:
        h = gf_mul(h, g, p, K)
        h = gf_add_ground(h, c, p, K)
    return h