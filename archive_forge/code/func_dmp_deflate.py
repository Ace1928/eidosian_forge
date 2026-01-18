from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_deflate(f, u, K):
    """
    Map ``x_i**m_i`` to ``y_i`` in a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_deflate

    >>> f = ZZ.map([[1, 0, 0, 2], [], [3, 0, 0, 4]])

    >>> dmp_deflate(f, 1, ZZ)
    ((2, 3), [[1, 2], [3, 4]])

    """
    if dmp_zero_p(f, u):
        return ((1,) * (u + 1), f)
    F = dmp_to_dict(f, u)
    B = [0] * (u + 1)
    for M in F.keys():
        for i, m in enumerate(M):
            B[i] = igcd(B[i], m)
    for i, b in enumerate(B):
        if not b:
            B[i] = 1
    B = tuple(B)
    if all((b == 1 for b in B)):
        return (B, f)
    H = {}
    for A, coeff in F.items():
        N = [a // b for a, b in zip(A, B)]
        H[tuple(N)] = coeff
    return (B, dmp_from_dict(H, u, K))