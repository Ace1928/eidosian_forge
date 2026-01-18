from sympy.polys.monomials import (
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.abc import a, b, c, x, y, z
from sympy.core import S, symbols
from sympy.testing.pytest import raises
def test_Monomial():
    m = Monomial((3, 4, 1), (x, y, z))
    n = Monomial((1, 2, 0), (x, y, z))
    assert m.as_expr() == x ** 3 * y ** 4 * z
    assert n.as_expr() == x ** 1 * y ** 2
    assert m.as_expr(a, b, c) == a ** 3 * b ** 4 * c
    assert n.as_expr(a, b, c) == a ** 1 * b ** 2
    assert m.exponents == (3, 4, 1)
    assert m.gens == (x, y, z)
    assert n.exponents == (1, 2, 0)
    assert n.gens == (x, y, z)
    assert m == (3, 4, 1)
    assert n != (3, 4, 1)
    assert m != (1, 2, 0)
    assert n == (1, 2, 0)
    assert (m == 1) is False
    assert m[0] == m[-3] == 3
    assert m[1] == m[-2] == 4
    assert m[2] == m[-1] == 1
    assert n[0] == n[-3] == 1
    assert n[1] == n[-2] == 2
    assert n[2] == n[-1] == 0
    assert m[:2] == (3, 4)
    assert n[:2] == (1, 2)
    assert m * n == Monomial((4, 6, 1))
    assert m / n == Monomial((2, 2, 1))
    assert m * (1, 2, 0) == Monomial((4, 6, 1))
    assert m / (1, 2, 0) == Monomial((2, 2, 1))
    assert m.gcd(n) == Monomial((1, 2, 0))
    assert m.lcm(n) == Monomial((3, 4, 1))
    assert m.gcd((1, 2, 0)) == Monomial((1, 2, 0))
    assert m.lcm((1, 2, 0)) == Monomial((3, 4, 1))
    assert m ** 0 == Monomial((0, 0, 0))
    assert m ** 1 == m
    assert m ** 2 == Monomial((6, 8, 2))
    assert m ** 3 == Monomial((9, 12, 3))
    raises(ExactQuotientFailed, lambda: m / Monomial((5, 2, 0)))
    mm = Monomial((1, 2, 3))
    raises(ValueError, lambda: mm.as_expr())
    assert str(mm) == 'Monomial((1, 2, 3))'
    assert str(m) == 'x**3*y**4*z**1'
    raises(NotImplementedError, lambda: m * 1)
    raises(NotImplementedError, lambda: m / 1)
    raises(ValueError, lambda: m ** (-1))
    raises(TypeError, lambda: m.gcd(3))
    raises(TypeError, lambda: m.lcm(3))