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
def test_issue_4825():
    raises(ValueError, lambda: dsolve(f(x, y).diff(x) - y * f(x, y), f(x)))
    assert classify_ode(f(x, y).diff(x) - y * f(x, y), f(x), dict=True) == {'order': 0, 'default': None, 'ordered_hints': ()}
    raises(ValueError, lambda: dsolve(f(x).diff(x), f(y)))
    assert classify_ode(f(x).diff(x), f(y), dict=True) == {'order': 0, 'default': None, 'ordered_hints': ()}