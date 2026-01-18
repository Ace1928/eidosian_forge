from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_zero_p(f, u):
    """
    Return ``True`` if ``f`` is zero in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_zero_p

    >>> dmp_zero_p([[[[[]]]]], 4)
    True
    >>> dmp_zero_p([[[[[1]]]]], 4)
    False

    """
    while u:
        if len(f) != 1:
            return False
        f = f[0]
        u -= 1
    return not f