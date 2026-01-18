from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_terms_gcd(f, u, K):
    """
    Remove GCD of terms from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_terms_gcd

    >>> f = ZZ.map([[1, 0], [1, 0, 0], [], []])

    >>> dmp_terms_gcd(f, 1, ZZ)
    ((2, 1), [[1], [1, 0]])

    """
    if dmp_ground_TC(f, u, K) or dmp_zero_p(f, u):
        return ((0,) * (u + 1), f)
    F = dmp_to_dict(f, u)
    G = monomial_min(*list(F.keys()))
    if all((g == 0 for g in G)):
        return (G, f)
    f = {}
    for monom, coeff in F.items():
        f[monomial_div(monom, G)] = coeff
    return (G, dmp_from_dict(f, u, K))