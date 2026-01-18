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
def test_issue_3505():
    e = sin(x) ** (-4) * (sqrt(cos(x)) * sin(x) ** 2 - cos(x) ** Rational(1, 3) * sin(x) ** 2)
    assert e.nseries(x, n=9) == Rational(-1, 12) - 7 * x ** 2 / 288 - 43 * x ** 4 / 10368 - 1123 * x ** 6 / 2488320 + 377 * x ** 8 / 29859840 + O(x ** 9)