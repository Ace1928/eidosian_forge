from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_rshift(f, n, K):
    """
    Efficiently divide ``f`` by ``x**n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_rshift

    >>> gf_rshift([1, 2, 3, 4, 0], 3, ZZ)
    ([1, 2], [3, 4, 0])

    """
    if not n:
        return (f, [])
    else:
        return (f[:-n], f[-n:])