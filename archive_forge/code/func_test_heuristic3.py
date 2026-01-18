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
def test_heuristic3():
    a, b = symbols('a b')
    df = f(x).diff(x)
    eq = x ** 2 * df + x * f(x) + f(x) ** 2 + x ** 2
    i = infinitesimals(eq, hint='bivariate')
    assert i == [{eta(x, f(x)): f(x), xi(x, f(x)): x}]
    assert checkinfsol(eq, i)[0]
    eq = x ** 2 * (-f(x) ** 2 + df) - a * x ** 2 * f(x) + 2 - a * x
    i = infinitesimals(eq, hint='bivariate')
    assert checkinfsol(eq, i)[0]