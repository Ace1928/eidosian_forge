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
def test_csch():
    x, y = symbols('x,y')
    k = Symbol('k', integer=True)
    n = Symbol('n', positive=True)
    assert csch(nan) is nan
    assert csch(zoo) is nan
    assert csch(oo) == 0
    assert csch(-oo) == 0
    assert csch(0) is zoo
    assert csch(-1) == -csch(1)
    assert csch(-x) == -csch(x)
    assert csch(-pi) == -csch(pi)
    assert csch(-2 ** 1024 * E) == -csch(2 ** 1024 * E)
    assert csch(pi * I) is zoo
    assert csch(-pi * I) is zoo
    assert csch(2 * pi * I) is zoo
    assert csch(-2 * pi * I) is zoo
    assert csch(-3 * 10 ** 73 * pi * I) is zoo
    assert csch(7 * 10 ** 103 * pi * I) is zoo
    assert csch(pi * I / 2) == -I
    assert csch(-pi * I / 2) == I
    assert csch(pi * I * Rational(5, 2)) == -I
    assert csch(pi * I * Rational(7, 2)) == I
    assert csch(pi * I / 3) == -2 / sqrt(3) * I
    assert csch(pi * I * Rational(-2, 3)) == 2 / sqrt(3) * I
    assert csch(pi * I / 4) == -sqrt(2) * I
    assert csch(-pi * I / 4) == sqrt(2) * I
    assert csch(pi * I * Rational(7, 4)) == sqrt(2) * I
    assert csch(pi * I * Rational(-3, 4)) == sqrt(2) * I
    assert csch(pi * I / 6) == -2 * I
    assert csch(-pi * I / 6) == 2 * I
    assert csch(pi * I * Rational(7, 6)) == 2 * I
    assert csch(pi * I * Rational(-7, 6)) == -2 * I
    assert csch(pi * I * Rational(-5, 6)) == 2 * I
    assert csch(pi * I / 105) == -1 / sin(pi / 105) * I
    assert csch(-pi * I / 105) == 1 / sin(pi / 105) * I
    assert csch(x * I) == -1 / sin(x) * I
    assert csch(k * pi * I) is zoo
    assert csch(17 * k * pi * I) is zoo
    assert csch(k * pi * I / 2) == -1 / sin(k * pi / 2) * I
    assert csch(n).is_real is True
    assert expand_trig(csch(x + y)) == 1 / (sinh(x) * cosh(y) + cosh(x) * sinh(y))