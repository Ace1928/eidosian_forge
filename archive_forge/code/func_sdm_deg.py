from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_deg(f):
    """
    Degree of ``f``.

    This is the maximum of the degrees of all its monomials.
    Invalid if ``f`` is zero.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_deg
    >>> sdm_deg([((1, 2, 3), 1), ((10, 0, 1), 1), ((2, 3, 4), 4)])
    7
    """
    return max((sdm_monomial_deg(M[0]) for M in f))