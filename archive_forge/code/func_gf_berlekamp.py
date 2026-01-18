from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_berlekamp(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for small ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_berlekamp

    >>> gf_berlekamp([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 2], [1, 0, 3]]

    """
    Q = gf_Qmatrix(f, p, K)
    V = gf_Qbasis(Q, p, K)
    for i, v in enumerate(V):
        V[i] = gf_strip(list(reversed(v)))
    factors = [f]
    for k in range(1, len(V)):
        for f in list(factors):
            s = K.zero
            while s < p:
                g = gf_sub_ground(V[k], s, p, K)
                h = gf_gcd(f, g, p, K)
                if h != [K.one] and h != f:
                    factors.remove(f)
                    f = gf_quo(f, h, p, K)
                    factors.extend([f, h])
                if len(factors) == len(V):
                    return _sort_factors(factors, multiple=False)
                s += K.one
    return _sort_factors(factors, multiple=False)