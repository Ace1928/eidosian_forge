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
def test_issue_3224():
    f = sqrt(1 - sqrt(y))
    assert f.nseries(y, 0, 2) == 1 - sqrt(y) / 2 - y / 8 - sqrt(y) ** 3 / 16 + O(y ** 2)