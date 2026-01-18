from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
from sympy.core.numbers import (E, I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, XFAIL
def test_expbug4():
    x = Symbol('x', real=True)
    assert (log(sin(2 * x) / x) * (1 + x)).series(x, 0, 2) == log(2) + x * log(2) + O(x ** 2, x)
    assert exp(log(sin(2 * x) / x) * (1 + x)).series(x, 0, 2) == 2 + 2 * x * log(2) + O(x ** 2)
    assert exp(log(2) + O(x)).nseries(x, 0, 2) == 2 + O(x)
    assert ((2 + O(x)) ** (1 + x)).nseries(x, 0, 2) == 2 + O(x)