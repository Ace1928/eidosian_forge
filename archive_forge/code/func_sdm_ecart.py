from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_ecart(f):
    """
    Compute the ecart of ``f``.

    This is defined to be the difference of the total degree of `f` and the
    total degree of the leading monomial of `f` [SCA, defn 2.3.7].

    Invalid if f is zero.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_ecart
    >>> sdm_ecart([((1, 2, 3), 1), ((1, 0, 1), 1)])
    0
    >>> sdm_ecart([((2, 2, 1), 1), ((1, 5, 1), 1)])
    3
    """
    return sdm_deg(f) - sdm_monomial_deg(sdm_LM(f))