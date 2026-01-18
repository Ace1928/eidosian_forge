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
@XFAIL
def test_kamke():
    a, b, alpha, c = symbols('a b alpha c')
    eq = x ** 2 * (a * f(x) ** 2 + f(x).diff(x)) + b * x ** alpha + c
    i = infinitesimals(eq, hint='sum_function')
    assert checkinfsol(eq, i)[0]