from sympy.core.numbers import Rational as Q
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.orthopolys import (
from sympy.abc import x, a, b
def test_legendre_poly():
    raises(ValueError, lambda: legendre_poly(-1, x))
    assert legendre_poly(1, x, polys=True) == Poly(x, domain='QQ')
    assert legendre_poly(0, x) == 1
    assert legendre_poly(1, x) == x
    assert legendre_poly(2, x) == Q(3, 2) * x ** 2 - Q(1, 2)
    assert legendre_poly(3, x) == Q(5, 2) * x ** 3 - Q(3, 2) * x
    assert legendre_poly(4, x) == Q(35, 8) * x ** 4 - Q(30, 8) * x ** 2 + Q(3, 8)
    assert legendre_poly(5, x) == Q(63, 8) * x ** 5 - Q(70, 8) * x ** 3 + Q(15, 8) * x
    assert legendre_poly(6, x) == Q(231, 16) * x ** 6 - Q(315, 16) * x ** 4 + Q(105, 16) * x ** 2 - Q(5, 16)
    assert legendre_poly(1).dummy_eq(x)
    assert legendre_poly(1, polys=True) == Poly(x)