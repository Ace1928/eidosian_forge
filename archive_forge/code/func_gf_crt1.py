from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_crt1(M, K):
    """
    First part of the Chinese Remainder Theorem.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_crt1

    >>> gf_crt1([99, 97, 95], ZZ)
    (912285, [9215, 9405, 9603], [62, 24, 12])

    """
    E, S = ([], [])
    p = prod(M, start=K.one)
    for m in M:
        E.append(p // m)
        S.append(K.gcdex(E[-1], m)[0] % m)
    return (p, E, S)