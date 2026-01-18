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
def test_issue_22604():
    x1, x2 = symbols('x1, x2', cls=Function)
    t, k1, k2, m1, m2 = symbols('t k1 k2 m1 m2', real=True)
    k1, k2, m1, m2 = (1, 1, 1, 1)
    eq1 = Eq(m1 * diff(x1(t), t, 2) + k1 * x1(t) - k2 * (x2(t) - x1(t)), 0)
    eq2 = Eq(m2 * diff(x2(t), t, 2) + k2 * (x2(t) - x1(t)), 0)
    eqs = [eq1, eq2]
    [x1sol, x2sol] = dsolve(eqs, [x1(t), x2(t)], ics={x1(0): 0, x1(t).diff().subs(t, 0): 0, x2(0): 1, x2(t).diff().subs(t, 0): 0})
    assert x1sol == Eq(x1(t), sqrt(3 - sqrt(5)) * (sqrt(10) + 5 * sqrt(2)) * cos(sqrt(2) * t * sqrt(3 - sqrt(5)) / 2) / 20 + (-5 * sqrt(2) + sqrt(10)) * sqrt(sqrt(5) + 3) * cos(sqrt(2) * t * sqrt(sqrt(5) + 3) / 2) / 20)
    assert x2sol == Eq(x2(t), (sqrt(5) + 5) * cos(sqrt(2) * t * sqrt(3 - sqrt(5)) / 2) / 10 + (5 - sqrt(5)) * cos(sqrt(2) * t * sqrt(sqrt(5) + 3) / 2) / 10)