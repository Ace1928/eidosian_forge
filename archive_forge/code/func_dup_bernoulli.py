from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_bernoulli(n, K):
    """Low-level implementation of Bernoulli polynomials."""
    if n < 1:
        return [K.one]
    p = [K.one, K(-1, 2)]
    for i in range(2, n + 1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K(1, 2), K) * K(1 << i - 1, (1 << i) - 1), K)
    return p