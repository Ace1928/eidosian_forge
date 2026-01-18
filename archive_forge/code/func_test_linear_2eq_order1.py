from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
def test_linear_2eq_order1():
    x, y, z = symbols('x, y, z', cls=Function)
    k, l, m, n = symbols('k, l, m, n', Integer=True)
    t = Symbol('t')
    x0, y0 = symbols('x0, y0', cls=Function)
    eq1 = (Eq(diff(x(t), t), x(t) + y(t) + 9), Eq(diff(y(t), t), 2 * x(t) + 5 * y(t) + 23))
    sol1 = [Eq(x(t), C1 * exp(t * (sqrt(6) + 3)) + C2 * exp(t * (-sqrt(6) + 3)) - Rational(22, 3)), Eq(y(t), C1 * (2 + sqrt(6)) * exp(t * (sqrt(6) + 3)) + C2 * (-sqrt(6) + 2) * exp(t * (-sqrt(6) + 3)) - Rational(5, 3))]
    assert checksysodesol(eq1, sol1) == (True, [0, 0])
    eq2 = (Eq(diff(x(t), t), x(t) + y(t) + 81), Eq(diff(y(t), t), -2 * x(t) + y(t) + 23))
    sol2 = [Eq(x(t), (C1 * cos(sqrt(2) * t) + C2 * sin(sqrt(2) * t)) * exp(t) - Rational(58, 3)), Eq(y(t), (-sqrt(2) * C1 * sin(sqrt(2) * t) + sqrt(2) * C2 * cos(sqrt(2) * t)) * exp(t) - Rational(185, 3))]
    assert checksysodesol(eq2, sol2) == (True, [0, 0])
    eq3 = (Eq(diff(x(t), t), 5 * t * x(t) + 2 * y(t)), Eq(diff(y(t), t), 2 * x(t) + 5 * t * y(t)))
    sol3 = [Eq(x(t), (C1 * exp(2 * t) + C2 * exp(-2 * t)) * exp(Rational(5, 2) * t ** 2)), Eq(y(t), (C1 * exp(2 * t) - C2 * exp(-2 * t)) * exp(Rational(5, 2) * t ** 2))]
    assert checksysodesol(eq3, sol3) == (True, [0, 0])
    eq4 = (Eq(diff(x(t), t), 5 * t * x(t) + t ** 2 * y(t)), Eq(diff(y(t), t), -t ** 2 * x(t) + 5 * t * y(t)))
    sol4 = [Eq(x(t), (C1 * cos(t ** 3 / 3) + C2 * sin(t ** 3 / 3)) * exp(Rational(5, 2) * t ** 2)), Eq(y(t), (-C1 * sin(t ** 3 / 3) + C2 * cos(t ** 3 / 3)) * exp(Rational(5, 2) * t ** 2))]
    assert checksysodesol(eq4, sol4) == (True, [0, 0])
    eq5 = (Eq(diff(x(t), t), 5 * t * x(t) + t ** 2 * y(t)), Eq(diff(y(t), t), -t ** 2 * x(t) + (5 * t + 9 * t ** 2) * y(t)))
    sol5 = [Eq(x(t), (C1 * exp((sqrt(77) / 2 + Rational(9, 2)) * t ** 3 / 3) + C2 * exp((-sqrt(77) / 2 + Rational(9, 2)) * t ** 3 / 3)) * exp(Rational(5, 2) * t ** 2)), Eq(y(t), (C1 * (sqrt(77) / 2 + Rational(9, 2)) * exp((sqrt(77) / 2 + Rational(9, 2)) * t ** 3 / 3) + C2 * (-sqrt(77) / 2 + Rational(9, 2)) * exp((-sqrt(77) / 2 + Rational(9, 2)) * t ** 3 / 3)) * exp(Rational(5, 2) * t ** 2))]
    assert checksysodesol(eq5, sol5) == (True, [0, 0])
    eq6 = (Eq(diff(x(t), t), 5 * t * x(t) + t ** 2 * y(t)), Eq(diff(y(t), t), (1 - t ** 2) * x(t) + (5 * t + 9 * t ** 2) * y(t)))
    sol6 = [Eq(x(t), C1 * x0(t) + C2 * x0(t) * Integral(t ** 2 * exp(Integral(5 * t, t)) * exp(Integral(9 * t ** 2 + 5 * t, t)) / x0(t) ** 2, t)), Eq(y(t), C1 * y0(t) + C2 * (y0(t) * Integral(t ** 2 * exp(Integral(5 * t, t)) * exp(Integral(9 * t ** 2 + 5 * t, t)) / x0(t) ** 2, t) + exp(Integral(5 * t, t)) * exp(Integral(9 * t ** 2 + 5 * t, t)) / x0(t)))]
    s = dsolve(eq6)
    assert s == sol6