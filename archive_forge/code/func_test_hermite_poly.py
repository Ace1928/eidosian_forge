from sympy.core.numbers import Rational as Q
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.orthopolys import (
from sympy.abc import x, a, b
def test_hermite_poly():
    raises(ValueError, lambda: hermite_poly(-1, x))
    assert hermite_poly(1, x, polys=True) == Poly(2 * x)
    assert hermite_poly(0, x) == 1
    assert hermite_poly(1, x) == 2 * x
    assert hermite_poly(2, x) == 4 * x ** 2 - 2
    assert hermite_poly(3, x) == 8 * x ** 3 - 12 * x
    assert hermite_poly(4, x) == 16 * x ** 4 - 48 * x ** 2 + 12
    assert hermite_poly(5, x) == 32 * x ** 5 - 160 * x ** 3 + 120 * x
    assert hermite_poly(6, x) == 64 * x ** 6 - 480 * x ** 4 + 720 * x ** 2 - 120
    assert hermite_poly(1).dummy_eq(2 * x)
    assert hermite_poly(1, polys=True) == Poly(2 * x)