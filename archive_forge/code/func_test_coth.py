from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.function import (expand_mul, expand_trig)
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
def test_coth():
    x, y = symbols('x,y')
    k = Symbol('k', integer=True)
    assert coth(nan) is nan
    assert coth(zoo) is nan
    assert coth(oo) == 1
    assert coth(-oo) == -1
    assert coth(0) is zoo
    assert unchanged(coth, 1)
    assert coth(-1) == -coth(1)
    assert unchanged(coth, x)
    assert coth(-x) == -coth(x)
    assert coth(pi * I) == -I * cot(pi)
    assert coth(-pi * I) == cot(pi) * I
    assert unchanged(coth, 2 ** 1024 * E)
    assert coth(-2 ** 1024 * E) == -coth(2 ** 1024 * E)
    assert coth(pi * I) == -I * cot(pi)
    assert coth(-pi * I) == I * cot(pi)
    assert coth(2 * pi * I) == -I * cot(2 * pi)
    assert coth(-2 * pi * I) == I * cot(2 * pi)
    assert coth(-3 * 10 ** 73 * pi * I) == I * cot(3 * 10 ** 73 * pi)
    assert coth(7 * 10 ** 103 * pi * I) == -I * cot(7 * 10 ** 103 * pi)
    assert coth(pi * I / 2) == 0
    assert coth(-pi * I / 2) == 0
    assert coth(pi * I * Rational(5, 2)) == 0
    assert coth(pi * I * Rational(7, 2)) == 0
    assert coth(pi * I / 3) == -I / sqrt(3)
    assert coth(pi * I * Rational(-2, 3)) == -I / sqrt(3)
    assert coth(pi * I / 4) == -I
    assert coth(-pi * I / 4) == I
    assert coth(pi * I * Rational(17, 4)) == -I
    assert coth(pi * I * Rational(-3, 4)) == -I
    assert coth(pi * I / 6) == -sqrt(3) * I
    assert coth(-pi * I / 6) == sqrt(3) * I
    assert coth(pi * I * Rational(7, 6)) == -sqrt(3) * I
    assert coth(pi * I * Rational(-5, 6)) == -sqrt(3) * I
    assert coth(pi * I / 105) == -cot(pi / 105) * I
    assert coth(-pi * I / 105) == cot(pi / 105) * I
    assert unchanged(coth, 2 + 3 * I)
    assert coth(x * I) == -cot(x) * I
    assert coth(k * pi * I) == -cot(k * pi) * I
    assert coth(17 * k * pi * I) == -cot(17 * k * pi) * I
    assert coth(k * pi * I) == -cot(k * pi) * I
    assert coth(log(tan(2))) == coth(log(-tan(2)))
    assert coth(1 + I * pi / 2) == tanh(1)
    assert coth(x).as_real_imag(deep=False) == (sinh(re(x)) * cosh(re(x)) / (sin(im(x)) ** 2 + sinh(re(x)) ** 2), -sin(im(x)) * cos(im(x)) / (sin(im(x)) ** 2 + sinh(re(x)) ** 2))
    x = Symbol('x', extended_real=True)
    assert coth(x).as_real_imag(deep=False) == (coth(x), 0)
    assert expand_trig(coth(2 * x)) == (coth(x) ** 2 + 1) / (2 * coth(x))
    assert expand_trig(coth(3 * x)) == (coth(x) ** 3 + 3 * coth(x)) / (1 + 3 * coth(x) ** 2)
    assert expand_trig(coth(x + y)) == (1 + coth(x) * coth(y)) / (coth(x) + coth(y))