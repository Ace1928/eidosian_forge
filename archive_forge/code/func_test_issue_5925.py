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
def test_issue_5925():
    sx = sqrt(x + z).series(z, 0, 1)
    sxy = sqrt(x + y + z).series(z, 0, 1)
    s1, s2 = (sx.subs(x, x + y), sxy)
    assert (s1 - s2).expand().removeO().simplify() == 0
    sx = sqrt(x + z).series(z, 0, 1)
    sxy = sqrt(x + y + z).series(z, 0, 1)
    assert sxy.subs({x: 1, y: 2}) == sx.subs(x, 3)