from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sech, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import (Integral, integrate)
from sympy.testing.pytest import XFAIL, SKIP, slow, skip, ON_CI
from sympy.abc import x, k, c, y, b, h, a, m, z, n, t
@XFAIL
@slow
def test_issue_4941():
    if ON_CI:
        skip('Too slow for CI.')
    assert not integrate(sqrt(1 + sinh(x / 20) ** 2), (x, -25, 25)).has(Integral)