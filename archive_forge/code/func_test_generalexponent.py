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
def test_generalexponent():
    p = 2
    e = (2 / x + 3 / x ** p) / (1 / x + 1 / x ** p)
    assert e.nseries(x, 0, 3) == 3 - x + x ** 2 + O(x ** 3)
    p = S.Half
    e = (2 / x + 3 / x ** p) / (1 / x + 1 / x ** p)
    assert e.nseries(x, 0, 2) == 2 - x + sqrt(x) + x ** (S(3) / 2) + O(x ** 2)
    e = 1 + sqrt(x)
    assert e.nseries(x, 0, 4) == 1 + sqrt(x)