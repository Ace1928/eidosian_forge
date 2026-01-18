from sympy.polys.monomials import (
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.abc import a, b, c, x, y, z
from sympy.core import S, symbols
from sympy.testing.pytest import raises
def test_monomial_divides():
    assert monomial_divides((1, 2, 3), (4, 5, 6)) is True
    assert monomial_divides((1, 2, 3), (0, 5, 6)) is False