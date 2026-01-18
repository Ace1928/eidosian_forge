from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def removethis(pair):
    i, j, s, t = pair
    if LMf[0] != t[0]:
        return False
    tik = sdm_monomial_lcm(LMf, sdm_LM(S[i]))
    tjk = sdm_monomial_lcm(LMf, sdm_LM(S[j]))
    return tik != t and tjk != t and sdm_monomial_divides(tik, t) and sdm_monomial_divides(tjk, t)