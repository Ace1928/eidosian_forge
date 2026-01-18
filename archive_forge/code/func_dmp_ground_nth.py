from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_ground_nth(f, N, u, K):
    """
    Return the ground ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_nth

    >>> f = ZZ.map([[1], [2, 3]])

    >>> dmp_ground_nth(f, (0, 1), 1, ZZ)
    2

    """
    v = u
    for n in N:
        if n < 0:
            raise IndexError('`n` must be non-negative, got %i' % n)
        elif n >= len(f):
            return K.zero
        else:
            d = dmp_degree(f, v)
            if d == -oo:
                d = -1
            f, v = (f[d - n], v - 1)
    return f