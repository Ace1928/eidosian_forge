from sympy.core.numbers import Rational as Q
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.orthopolys import (
from sympy.abc import x, a, b
def test_laguerre_poly():
    raises(ValueError, lambda: laguerre_poly(-1, x))
    assert laguerre_poly(1, x, polys=True) == Poly(-x + 1, domain='QQ')
    assert laguerre_poly(0, x) == 1
    assert laguerre_poly(1, x) == -x + 1
    assert laguerre_poly(2, x) == Q(1, 2) * x ** 2 - Q(4, 2) * x + 1
    assert laguerre_poly(3, x) == -Q(1, 6) * x ** 3 + Q(9, 6) * x ** 2 - Q(18, 6) * x + 1
    assert laguerre_poly(4, x) == Q(1, 24) * x ** 4 - Q(16, 24) * x ** 3 + Q(72, 24) * x ** 2 - Q(96, 24) * x + 1
    assert laguerre_poly(5, x) == -Q(1, 120) * x ** 5 + Q(25, 120) * x ** 4 - Q(200, 120) * x ** 3 + Q(600, 120) * x ** 2 - Q(600, 120) * x + 1
    assert laguerre_poly(6, x) == Q(1, 720) * x ** 6 - Q(36, 720) * x ** 5 + Q(450, 720) * x ** 4 - Q(2400, 720) * x ** 3 + Q(5400, 720) * x ** 2 - Q(4320, 720) * x + 1
    assert laguerre_poly(0, x, a) == 1
    assert laguerre_poly(1, x, a) == -x + a + 1
    assert laguerre_poly(2, x, a) == x ** 2 / 2 + (-a - 2) * x + a ** 2 / 2 + a * Q(3, 2) + 1
    assert laguerre_poly(3, x, a) == -x ** 3 / 6 + (a / 2 + Q(3) / 2) * x ** 2 + (-a ** 2 / 2 - a * Q(5, 2) - 3) * x + a ** 3 / 6 + a ** 2 + a * Q(11, 6) + 1
    assert laguerre_poly(1).dummy_eq(-x + 1)
    assert laguerre_poly(1, polys=True) == Poly(-x + 1)