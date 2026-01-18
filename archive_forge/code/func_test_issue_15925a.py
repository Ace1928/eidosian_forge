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
def test_issue_15925a():
    assert not integrate(sqrt((1 + sin(x)) ** 2 + cos(x) ** 2), (x, -pi / 2, pi / 2)).has(Integral)