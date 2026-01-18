from sympy.core.numbers import Rational as Q
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.orthopolys import (
from sympy.abc import x, a, b
def test_gegenbauer_poly():
    raises(ValueError, lambda: gegenbauer_poly(-1, a, x))
    assert gegenbauer_poly(1, a, x, polys=True) == Poly(2 * a * x, x, domain='ZZ(a)')
    assert gegenbauer_poly(0, a, x) == 1
    assert gegenbauer_poly(1, a, x) == 2 * a * x
    assert gegenbauer_poly(2, a, x) == -a + x ** 2 * (2 * a ** 2 + 2 * a)
    assert gegenbauer_poly(3, a, x) == x ** 3 * (4 * a ** 3 / 3 + 4 * a ** 2 + a * Q(8, 3)) + x * (-2 * a ** 2 - 2 * a)
    assert gegenbauer_poly(1, S.Half).dummy_eq(x)
    assert gegenbauer_poly(1, a, polys=True) == Poly(2 * a * x, x, domain='ZZ(a)')