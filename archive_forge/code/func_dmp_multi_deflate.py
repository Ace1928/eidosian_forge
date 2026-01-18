from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_multi_deflate(polys, u, K):
    """
    Map ``x_i**m_i`` to ``y_i`` in a set of polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_multi_deflate

    >>> f = ZZ.map([[1, 0, 0, 2], [], [3, 0, 0, 4]])
    >>> g = ZZ.map([[1, 0, 2], [], [3, 0, 4]])

    >>> dmp_multi_deflate((f, g), 1, ZZ)
    ((2, 1), ([[1, 0, 0, 2], [3, 0, 0, 4]], [[1, 0, 2], [3, 0, 4]]))

    """
    if not u:
        M, H = dup_multi_deflate(polys, K)
        return ((M,), H)
    F, B = ([], [0] * (u + 1))
    for p in polys:
        f = dmp_to_dict(p, u)
        if not dmp_zero_p(p, u):
            for M in f.keys():
                for i, m in enumerate(M):
                    B[i] = igcd(B[i], m)
        F.append(f)
    for i, b in enumerate(B):
        if not b:
            B[i] = 1
    B = tuple(B)
    if all((b == 1 for b in B)):
        return (B, polys)
    H = []
    for f in F:
        h = {}
        for A, coeff in f.items():
            N = [a // b for a, b in zip(A, B)]
            h[tuple(N)] = coeff
        H.append(dmp_from_dict(h, u, K))
    return (B, tuple(H))