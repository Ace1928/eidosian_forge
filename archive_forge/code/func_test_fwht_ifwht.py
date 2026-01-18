from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Symbol, symbols, I, Rational
from sympy.discrete import (fft, ifft, ntt, intt, fwht, ifwht,
from sympy.testing.pytest import raises
def test_fwht_ifwht():
    assert all((tf(ls) == ls for tf in (fwht, ifwht) for ls in ([], [Rational(7, 4)])))
    ls = [213, 321, 43235, 5325, 312, 53]
    fls = [49459, 38061, -47661, -37759, 48729, 37543, -48391, -38277]
    assert fwht(ls) == fls
    assert ifwht(fls) == ls + [S.Zero] * 2
    ls = [S.Half + 2 * I, Rational(3, 7) + 4 * I, Rational(5, 6) + 6 * I, Rational(7, 3), Rational(9, 4)]
    ifls = [Rational(533, 672) + I * 3 / 2, Rational(23, 224) + I / 2, Rational(1, 672), Rational(107, 224) - I, Rational(155, 672) + I * 3 / 2, Rational(-103, 224) + I / 2, Rational(-377, 672), Rational(-19, 224) - I]
    assert ifwht(ls) == ifls
    assert fwht(ifls) == ls + [S.Zero] * 3
    x, y = symbols('x y')
    raises(TypeError, lambda: fwht(x))
    ls = [x, 2 * x, 3 * x ** 2, 4 * x ** 3]
    ifls = [x ** 3 + 3 * x ** 2 / 4 + x * Rational(3, 4), -x ** 3 + 3 * x ** 2 / 4 - x / 4, -x ** 3 - 3 * x ** 2 / 4 + x * Rational(3, 4), x ** 3 - 3 * x ** 2 / 4 - x / 4]
    assert ifwht(ls) == ifls
    assert fwht(ifls) == ls
    ls = [x, y, x ** 2, y ** 2, x * y]
    fls = [x ** 2 + x * y + x + y ** 2 + y, x ** 2 + x * y + x - y ** 2 - y, -x ** 2 + x * y + x - y ** 2 + y, -x ** 2 + x * y + x + y ** 2 - y, x ** 2 - x * y + x + y ** 2 + y, x ** 2 - x * y + x - y ** 2 - y, -x ** 2 - x * y + x - y ** 2 + y, -x ** 2 - x * y + x + y ** 2 - y]
    assert fwht(ls) == fls
    assert ifwht(fls) == ls + [S.Zero] * 3
    ls = list(range(6))
    assert fwht(ls) == [x * 8 for x in ifwht(ls)]