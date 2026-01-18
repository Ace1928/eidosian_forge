from sympy.core.numbers import Rational as Q
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.appellseqs import (bernoulli_poly, bernoulli_c_poly,
from sympy.abc import x
def test_euler_poly():
    raises(ValueError, lambda: euler_poly(-1, x))
    assert euler_poly(1, x, polys=True) == Poly(x - Q(1, 2))
    assert euler_poly(0, x) == 1
    assert euler_poly(1, x) == x - Q(1, 2)
    assert euler_poly(2, x) == x ** 2 - x
    assert euler_poly(3, x) == x ** 3 - Q(3, 2) * x ** 2 + Q(1, 4)
    assert euler_poly(4, x) == x ** 4 - 2 * x ** 3 + x
    assert euler_poly(5, x) == x ** 5 - Q(5, 2) * x ** 4 + Q(5, 2) * x ** 2 - Q(1, 2)
    assert euler_poly(6, x) == x ** 6 - 3 * x ** 5 + 5 * x ** 3 - 3 * x
    assert euler_poly(1).dummy_eq(x - Q(1, 2))
    assert euler_poly(1, polys=True) == Poly(x - Q(1, 2))
    assert genocchi_poly(9, x) == euler_poly(8, x) * -9
    assert genocchi_poly(10, x) == euler_poly(9, x) * -10