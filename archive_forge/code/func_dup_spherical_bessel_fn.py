from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_spherical_bessel_fn(n, K):
    """Low-level implementation of fn(n, x)."""
    if n < 1:
        return [K.one, K.zero]
    m2, m1 = ([K.one], [K.one, K.zero])
    for i in range(2, n + 1):
        m2, m1 = (m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2 * i - 1), K), m2, K))
    return dup_lshift(m1, 1, K)