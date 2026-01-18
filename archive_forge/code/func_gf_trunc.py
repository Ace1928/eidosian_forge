from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_trunc(f, p):
    """
    Reduce all coefficients modulo ``p``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_trunc

    >>> gf_trunc([7, -2, 3], 5)
    [2, 3, 3]

    """
    return gf_strip([a % p for a in f])