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
def test_issue_3508():
    x = Symbol('x', real=True)
    assert log(sin(x)).series(x, n=5) == log(x) - x ** 2 / 6 - x ** 4 / 180 + O(x ** 5)
    e = -log(x) + x * (-log(x) + log(sin(2 * x))) + log(sin(2 * x))
    assert e.series(x, n=5) == log(2) + log(2) * x - 2 * x ** 2 / 3 - 2 * x ** 3 / 3 - 4 * x ** 4 / 45 + O(x ** 5)