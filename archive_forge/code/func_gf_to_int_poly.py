from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_to_int_poly(f, p, symmetric=True):
    """
    Convert a ``GF(p)[x]`` polynomial to ``Z[x]``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_to_int_poly

    >>> gf_to_int_poly([2, 3, 3], 5)
    [2, -2, -2]
    >>> gf_to_int_poly([2, 3, 3], 5, symmetric=False)
    [2, 3, 3]

    """
    if symmetric:
        return [gf_int(c, p) for c in f]
    else:
        return f