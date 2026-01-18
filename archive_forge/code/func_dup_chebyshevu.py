from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_chebyshevu(n, K):
    """Low-level implementation of Chebyshev polynomials of the second kind."""
    if n < 1:
        return [K.one]
    m2, m1 = ([K.one], [K(2), K.zero])
    for i in range(2, n + 1):
        m2, m1 = (m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K))
    return m1