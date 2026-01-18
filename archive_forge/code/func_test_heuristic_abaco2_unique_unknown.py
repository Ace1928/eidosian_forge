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
def test_heuristic_abaco2_unique_unknown():
    a, b = symbols('a b')
    F = Function('F')
    eq = f(x).diff(x) - x ** (a - 1) * f(x) ** (1 - b) * F(x ** a / a + f(x) ** b / b)
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert i == [{eta(x, f(x)): -f(x) * f(x) ** (-b), xi(x, f(x)): x * x ** (-a)}]
    assert checkinfsol(eq, i)[0]
    eq = f(x).diff(x) + tan(F(x ** 2 + f(x) ** 2) + atan(x / f(x)))
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert i == [{eta(x, f(x)): x, xi(x, f(x)): -f(x)}]
    assert checkinfsol(eq, i)[0]
    eq = (x * f(x).diff(x) + f(x) + 2 * x) ** 2 - 4 * x * f(x) - 4 * x ** 2 - 4 * a
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert checkinfsol(eq, i)[0]