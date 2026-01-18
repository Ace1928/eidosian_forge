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
def test_acsch_leading_term():
    x = Symbol('x')
    assert acsch(1 / x).as_leading_term(x) == x
    assert acsch(x + I).as_leading_term(x) == -I * pi / 2
    assert acsch(x - I).as_leading_term(x) == I * pi / 2
    assert acsch(x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert acsch(x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I * pi
    assert acsch(x + I / 2).as_leading_term(x, cdir=1) == -I * pi - acsch(I / 2)
    assert acsch(x + I / 2).as_leading_term(x, cdir=-1) == acsch(I / 2)
    assert acsch(x - I / 2).as_leading_term(x, cdir=1) == -acsch(I / 2)
    assert acsch(x - I / 2).as_leading_term(x, cdir=-1) == acsch(I / 2) + I * pi
    assert acsch(I / 2 + I * x - x ** 2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) - I * pi / 2
    assert acsch(I / 2 + I * x - x ** 2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) - I * pi / 2