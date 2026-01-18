from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_normal(f, K):
    """
    Normalize univariate polynomial in the given domain.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_normal

    >>> dup_normal([0, 1.5, 2, 3], ZZ)
    [1, 2, 3]

    """
    return dup_strip([K.normal(c) for c in f])