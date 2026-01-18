from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_eject(f, u, K, front=False):
    """
    Convert ``f`` from ``K[X,Y]`` to ``K[X][Y]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_eject

    >>> dmp_eject([[[1]], [[1], [2]]], 2, ZZ['x', 'y'])
    [1, x + 2]

    """
    f, h = (dmp_to_dict(f, u), {})
    n = K.ngens
    v = u - K.ngens + 1
    for monom, c in f.items():
        if front:
            g_monom, f_monom = (monom[:n], monom[n:])
        else:
            g_monom, f_monom = (monom[-n:], monom[:-n])
        if f_monom in h:
            h[f_monom][g_monom] = c
        else:
            h[f_monom] = {g_monom: c}
    for monom, c in h.items():
        h[monom] = K(c)
    return dmp_from_dict(h, v - 1, K)