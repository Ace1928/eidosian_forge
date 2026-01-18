from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_convolution_fwht():
    assert convolution_fwht([], []) == []
    assert convolution_fwht([], [1]) == []
    assert convolution_fwht([1, 2, 3], [4, 5, 6]) == [32, 13, 18, 27]
    assert convolution_fwht([Rational(5, 7), Rational(6, 8), Rational(7, 3)], [2, 4, Rational(6, 7)]) == [Rational(45, 7), Rational(61, 14), Rational(776, 147), Rational(419, 42)]
    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5), 4 + 5 * I]
    b = [94, 51, 53, 45, 31, 27, 13]
    c = [3 + 4 * I, 5 + 7 * I, 3, Rational(7, 6), 8]
    assert convolution_fwht(a, b) == [53 * sqrt(3) + 366 + 155 * I, 45 * sqrt(3) + Rational(5848, 15) + 135 * I, 94 * sqrt(3) + Rational(1257, 5) + 65 * I, 51 * sqrt(3) + Rational(3974, 15), 13 * sqrt(3) + 452 + 470 * I, Rational(4513, 15) + 255 * I, 31 * sqrt(3) + Rational(1314, 5) + 265 * I, 27 * sqrt(3) + Rational(3676, 15) + 225 * I]
    assert convolution_fwht(b, c) == [Rational(1993, 2) + 733 * I, Rational(6215, 6) + 862 * I, Rational(1659, 2) + 527 * I, Rational(1988, 3) + 551 * I, 1019 + 313 * I, Rational(3955, 6) + 325 * I, Rational(1175, 2) + 52 * I, Rational(3253, 6) + 91 * I]
    assert convolution_fwht(a[3:], c) == [Rational(-54, 5) + I * 293 / 5, -1 + I * 204 / 5, Rational(133, 15) + I * 35 / 6, Rational(409, 30) + 15 * I, Rational(56, 5), 32 + 40 * I, 0, 0]
    u, v, w, x, y, z = symbols('u v w x y z')
    assert convolution_fwht([u, v], [x, y]) == [u * x + v * y, u * y + v * x]
    assert convolution_fwht([u, v, w], [x, y]) == [u * x + v * y, u * y + v * x, w * x, w * y]
    assert convolution_fwht([u, v, w], [x, y, z]) == [u * x + v * y + w * z, u * y + v * x, u * z + w * x, v * z + w * y]
    raises(TypeError, lambda: convolution_fwht(x, y))
    raises(TypeError, lambda: convolution_fwht(x * y, u + v))