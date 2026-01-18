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
def test_asech_series():
    x = Symbol('x')
    assert asech(x).series(x, 0, 9, cdir=1) == log(2) - log(x) - x ** 2 / 4 - 3 * x ** 4 / 32 - 5 * x ** 6 / 96 - 35 * x ** 8 / 1024 + O(x ** 9)
    assert asech(x).series(x, 0, 9, cdir=-1) == I * pi + log(2) - log(-x) - x ** 2 / 4 - 3 * x ** 4 / 32 - 5 * x ** 6 / 96 - 35 * x ** 8 / 1024 + O(x ** 9)
    t6 = asech(x).taylor_term(6, x)
    assert t6 == -5 * x ** 6 / 96
    assert asech(x).taylor_term(8, x, t6, 0) == -35 * x ** 8 / 1024