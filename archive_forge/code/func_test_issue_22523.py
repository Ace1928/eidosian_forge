from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect
from sympy.solvers.ode import (classify_ode,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.utilities.misc import filldedent
def test_issue_22523():
    N, s = symbols('N s')
    rho = Function('rho')
    eqn = 4.0 * N * sqrt(N - 1) * rho(s) + (4 * s ** 2 * (N - 1) + (N - 2 * s * (N - 1)) ** 2) * Derivative(rho(s), (s, 2))
    match = classify_ode(eqn, dict=True, hint='all')
    assert match['2nd_power_series_ordinary']['terms'] == 5
    C1, C2 = symbols('C1,C2')
    sol = dsolve(eqn, hint='2nd_power_series_ordinary')
    assert filldedent(sol) == filldedent(str('\n        Eq(rho(s), C2*(1 - 4.0*s**4*sqrt(N - 1.0)/N + 0.666666666666667*s**4/N\n        - 2.66666666666667*s**3*sqrt(N - 1.0)/N - 2.0*s**2*sqrt(N - 1.0)/N +\n        9.33333333333333*s**4*sqrt(N - 1.0)/N**2 - 0.666666666666667*s**4/N**2\n        + 2.66666666666667*s**3*sqrt(N - 1.0)/N**2 -\n        5.33333333333333*s**4*sqrt(N - 1.0)/N**3) + C1*s*(1.0 -\n        1.33333333333333*s**3*sqrt(N - 1.0)/N - 0.666666666666667*s**2*sqrt(N\n        - 1.0)/N + 1.33333333333333*s**3*sqrt(N - 1.0)/N**2) + O(s**6))'))