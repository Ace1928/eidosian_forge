from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_sort(f, O):
    """Sort terms in ``f`` using the given monomial order ``O``. """
    return sorted(f, key=lambda term: O(term[0]), reverse=True)