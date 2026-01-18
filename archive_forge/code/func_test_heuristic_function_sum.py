from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.solvers.ode import (classify_ode, checkinfsol, dsolve, infinitesimals)
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import XFAIL
def test_heuristic_function_sum():
    eq = f(x).diff(x) - (3 * (1 + x ** 2 / f(x) ** 2) * atan(f(x) / x) + (1 - 2 * f(x)) / x + (1 - 3 * f(x)) * (x / f(x) ** 2))
    i = infinitesimals(eq, hint='function_sum')
    assert i == [{eta(x, f(x)): f(x) ** (-2) + x ** (-2), xi(x, f(x)): 0}]
    assert checkinfsol(eq, i)[0]