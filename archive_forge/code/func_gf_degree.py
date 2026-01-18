from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_degree(f):
    """
    Return the leading degree of ``f``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_degree

    >>> gf_degree([1, 1, 2, 0])
    3
    >>> gf_degree([])
    -1

    """
    return len(f) - 1