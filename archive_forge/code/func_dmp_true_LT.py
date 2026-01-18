from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_true_LT(f, u, K):
    """
    Return the leading term ``c * x_1**n_1 ... x_k**n_k``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_true_LT

    >>> f = ZZ.map([[4], [2, 0], [3, 0, 0]])

    >>> dmp_true_LT(f, 1, ZZ)
    ((2, 0), 4)

    """
    monom = []
    while u:
        monom.append(len(f) - 1)
        f, u = (f[0], u - 1)
    if not f:
        monom.append(0)
    else:
        monom.append(len(f) - 1)
    return (tuple(monom), dup_LC(f, K))