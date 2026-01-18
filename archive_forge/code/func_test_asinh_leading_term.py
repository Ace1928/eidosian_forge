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
def test_asinh_leading_term():
    x = Symbol('x')
    assert asinh(x).as_leading_term(x, cdir=1) == x
    assert asinh(x + I).as_leading_term(x, cdir=1) == I * pi / 2
    assert asinh(x - I).as_leading_term(x, cdir=1) == -I * pi / 2
    assert asinh(1 / x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert asinh(1 / x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I * pi
    assert asinh(x + 2 * I).as_leading_term(x, cdir=1) == I * asin(2)
    assert asinh(x + 2 * I).as_leading_term(x, cdir=-1) == -I * asin(2) + I * pi
    assert asinh(x - 2 * I).as_leading_term(x, cdir=1) == -I * pi + I * asin(2)
    assert asinh(x - 2 * I).as_leading_term(x, cdir=-1) == -I * asin(2)
    assert asinh(2 * I + I * x - x ** 2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) + I * pi / 2
    assert asinh(2 * I + I * x - x ** 2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) + I * pi / 2