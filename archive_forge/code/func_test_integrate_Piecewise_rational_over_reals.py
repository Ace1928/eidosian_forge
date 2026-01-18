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
def test_integrate_Piecewise_rational_over_reals():
    f = Piecewise((0, t - 478.515625 * pi < 0), (13.2075145209219 * pi / (0.000871222 * t + 0.995) ** 2, t - 478.515625 * pi >= 0))
    assert abs((integrate(f, (t, 0, oo)) - 15235.9375 * pi).evalf()) <= 1e-07