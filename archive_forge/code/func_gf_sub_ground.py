from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_sub_ground(f, a, p, K):
    """
    Compute ``f - a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub_ground

    >>> gf_sub_ground([3, 2, 4], 2, 5, ZZ)
    [3, 2, 2]

    """
    if not f:
        a = -a % p
    else:
        a = (f[-1] - a) % p
        if len(f) > 1:
            return f[:-1] + [a]
    if not a:
        return []
    else:
        return [a]