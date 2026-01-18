from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_from_sympy(f, K):
    """
    Convert the ground domain of ``f`` from SymPy to ``K``.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_sympy

    >>> dup_from_sympy([S(1), S(2)], ZZ) == [ZZ(1), ZZ(2)]
    True

    """
    return dup_strip([K.from_sympy(c) for c in f])