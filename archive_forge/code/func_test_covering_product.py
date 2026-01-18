from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_covering_product():
    assert covering_product([], []) == []
    assert covering_product([], [Rational(1, 3)]) == []
    assert covering_product([6 + I * 3 / 7], [Rational(2, 3)]) == [4 + I * 2 / 7]
    a = [1, Rational(5, 8), sqrt(7), 4 + 9 * I]
    b = [66, 81, 95, 49, 37, 89, 17]
    c = [3 + I * 2 / 3, 51 + 72 * I, 7, Rational(7, 15), 91]
    assert covering_product(a, b) == [66, Rational(1383, 8), 95 + 161 * sqrt(7), 130 * sqrt(7) + 1303 + 2619 * I, 37, Rational(671, 4), 17 + 54 * sqrt(7), 89 * sqrt(7) + Rational(4661, 8) + 1287 * I]
    assert covering_product(b, c) == [198 + 44 * I, 7740 + 10638 * I, 1412 + I * 190 / 3, Rational(42684, 5) + I * 31202 / 3, 9484 + I * 74 / 3, 22163 + I * 27394 / 3, 10621 + I * 34 / 3, Rational(90236, 15) + 1224 * I]
    assert covering_product(a, c) == covering_product(c, a)
    assert covering_product(b, c[:-1]) == [198 + 44 * I, 7740 + 10638 * I, 1412 + I * 190 / 3, Rational(42684, 5) + I * 31202 / 3, 111 + I * 74 / 3, 6693 + I * 27394 / 3, 429 + I * 34 / 3, Rational(23351, 15) + 1224 * I]
    assert covering_product(a, c[:-1]) == [3 + I * 2 / 3, Rational(339, 4) + I * 1409 / 12, 7 + 10 * sqrt(7) + 2 * sqrt(7) * I / 3, -403 + 772 * sqrt(7) / 15 + 72 * sqrt(7) * I + I * 12658 / 15]
    u, v, w, x, y, z = symbols('u v w x y z')
    assert covering_product([u, v, w], [x, y]) == [u * x, u * y + v * x + v * y, w * x, w * y]
    assert covering_product([u, v, w, x], [y, z]) == [u * y, u * z + v * y + v * z, w * y, w * z + x * y + x * z]
    assert covering_product([u, v], [x, y, z]) == covering_product([x, y, z], [u, v])
    raises(TypeError, lambda: covering_product(x, z))
    raises(TypeError, lambda: covering_product(Rational(7, 3), u))