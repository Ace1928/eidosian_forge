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
def test_homogeneous_order():
    assert homogeneous_order(exp(y / x) + tan(y / x), x, y) == 0
    assert homogeneous_order(x ** 2 + sin(x) * cos(y), x, y) is None
    assert homogeneous_order(x - y - x * sin(y / x), x, y) == 1
    assert homogeneous_order((x * y + sqrt(x ** 4 + y ** 4) + x ** 2 * (log(x) - log(y))) / (pi * x ** Rational(2, 3) * sqrt(y) ** 3), x, y) == Rational(-1, 6)
    assert homogeneous_order(y / x * cos(y / x) - x / y * sin(y / x) + cos(y / x), x, y) == 0
    assert homogeneous_order(f(x), x, f(x)) == 1
    assert homogeneous_order(f(x) ** 2, x, f(x)) == 2
    assert homogeneous_order(x * y * z, x, y) == 2
    assert homogeneous_order(x * y * z, x, y, z) == 3
    assert homogeneous_order(x ** 2 * f(x) / sqrt(x ** 2 + f(x) ** 2), f(x)) is None
    assert homogeneous_order(f(x, y) ** 2, x, f(x, y), y) == 2
    assert homogeneous_order(f(x, y) ** 2, x, f(x), y) is None
    assert homogeneous_order(f(x, y) ** 2, x, f(x, y)) is None
    assert homogeneous_order(f(y, x) ** 2, x, y, f(x, y)) is None
    assert homogeneous_order(f(y), f(x), x) is None
    assert homogeneous_order(-f(x) / x + 1 / sin(f(x) / x), f(x), x) == 0
    assert homogeneous_order(log(1 / y) + log(x ** 2), x, y) is None
    assert homogeneous_order(log(1 / y) + log(x), x, y) == 0
    assert homogeneous_order(log(x / y), x, y) == 0
    assert homogeneous_order(2 * log(1 / y) + 2 * log(x), x, y) == 0
    a = Symbol('a')
    assert homogeneous_order(a * log(1 / y) + a * log(x), x, y) == 0
    assert homogeneous_order(f(x).diff(x), x, y) is None
    assert homogeneous_order(-f(x).diff(x) + x, x, y) is None
    assert homogeneous_order(O(x), x, y) is None
    assert homogeneous_order(x + O(x ** 2), x, y) is None
    assert homogeneous_order(x ** pi, x) == pi
    assert homogeneous_order(x ** x, x) is None
    raises(ValueError, lambda: homogeneous_order(x * y))