from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_euler(n, K):
    """Low-level implementation of Euler polynomials."""
    return dup_quo_ground(dup_genocchi(n + 1, ZZ), K(-n - 1), K)