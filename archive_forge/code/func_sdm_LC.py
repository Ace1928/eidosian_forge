from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_LC(f, K):
    """Returns the leading coefficient of ``f``. """
    if not f:
        return K.zero
    else:
        return f[0][1]