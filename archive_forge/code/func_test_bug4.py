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
def test_bug4():
    w = Symbol('w')
    e = x / (w ** 4 + x ** 2 * w ** 4 + 2 * x * w ** 4) * w ** 4
    assert e.nseries(w, n=2).removeO().expand() in [x / (1 + 2 * x + x ** 2), 1 / (1 + x / 2 + 1 / x / 2) / 2, 1 / x / (1 + 2 / x + x ** (-2))]