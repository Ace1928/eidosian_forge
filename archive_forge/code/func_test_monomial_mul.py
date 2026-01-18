from sympy.polys.monomials import (
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.abc import a, b, c, x, y, z
from sympy.core import S, symbols
from sympy.testing.pytest import raises
def test_monomial_mul():
    assert monomial_mul((3, 4, 1), (1, 2, 0)) == (4, 6, 1)