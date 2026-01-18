from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_exquo(f, g, p, K):
    """
    Compute polynomial quotient in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_exquo

    >>> gf_exquo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)
    [3, 2, 4]

    >>> gf_exquo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [1, 1, 0] does not divide [1, 0, 1, 1]

    """
    q, r = gf_div(f, g, p, K)
    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)