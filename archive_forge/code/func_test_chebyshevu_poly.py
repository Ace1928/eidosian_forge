from sympy.core.numbers import Rational as Q
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.orthopolys import (
from sympy.abc import x, a, b
def test_chebyshevu_poly():
    raises(ValueError, lambda: chebyshevu_poly(-1, x))
    assert chebyshevu_poly(1, x, polys=True) == Poly(2 * x)
    assert chebyshevu_poly(0, x) == 1
    assert chebyshevu_poly(1, x) == 2 * x
    assert chebyshevu_poly(2, x) == 4 * x ** 2 - 1
    assert chebyshevu_poly(3, x) == 8 * x ** 3 - 4 * x
    assert chebyshevu_poly(4, x) == 16 * x ** 4 - 12 * x ** 2 + 1
    assert chebyshevu_poly(5, x) == 32 * x ** 5 - 32 * x ** 3 + 6 * x
    assert chebyshevu_poly(6, x) == 64 * x ** 6 - 80 * x ** 4 + 24 * x ** 2 - 1
    assert chebyshevu_poly(1).dummy_eq(2 * x)
    assert chebyshevu_poly(1, polys=True) == Poly(2 * x)