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
def test_acosh_series():
    x = Symbol('x')
    assert acosh(x).series(x, 0, 8) == -I * x + pi * I / 2 - I * x ** 3 / 6 - 3 * I * x ** 5 / 40 - 5 * I * x ** 7 / 112 + O(x ** 8)
    t5 = acosh(x).taylor_term(5, x)
    assert t5 == -3 * I * x ** 5 / 40
    assert acosh(x).taylor_term(7, x, t5, 0) == -5 * I * x ** 7 / 112