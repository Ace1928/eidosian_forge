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
def test_higher_order_to_first_order():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')
    eqs1 = [Eq(Derivative(f(x), (x, 2)), 2 * f(x) + g(x)), Eq(Derivative(g(x), (x, 2)), -f(x))]
    sol1 = [Eq(f(x), -C2 * x * exp(-x) + C3 * x * exp(x) - (C1 - C2) * exp(-x) + (C3 + C4) * exp(x)), Eq(g(x), C2 * x * exp(-x) - C3 * x * exp(x) + (C1 + C2) * exp(-x) + (C3 - C4) * exp(x))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])
    eqs2 = [Eq(f(x).diff(x, 2), 0), Eq(g(x).diff(x, 2), f(x))]
    sol2 = [Eq(f(x), C1 + C2 * x), Eq(g(x), C1 * x ** 2 / 2 + C2 * x ** 3 / 6 + C3 + C4 * x)]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])
    eqs3 = [Eq(Derivative(f(x), (x, 2)), 2 * f(x)), Eq(Derivative(g(x), (x, 2)), -f(x) + 2 * g(x))]
    sol3 = [Eq(f(x), 4 * C1 * exp(-sqrt(2) * x) + 4 * C2 * exp(sqrt(2) * x)), Eq(g(x), sqrt(2) * C1 * x * exp(-sqrt(2) * x) - sqrt(2) * C2 * x * exp(sqrt(2) * x) + (C1 + sqrt(2) * C4) * exp(-sqrt(2) * x) + (C2 - sqrt(2) * C3) * exp(sqrt(2) * x))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])
    eqs4 = [Eq(Derivative(f(x), (x, 2)), 2 * f(x) + g(x)), Eq(Derivative(g(x), (x, 2)), 2 * g(x))]
    sol4 = [Eq(f(x), C1 * x * exp(sqrt(2) * x) / 4 + C3 * x * exp(-sqrt(2) * x) / 4 + (C2 / 4 + sqrt(2) * C3 / 8) * exp(-sqrt(2) * x) - exp(sqrt(2) * x) * (sqrt(2) * C1 / 8 + C4 * Rational(-1, 4))), Eq(g(x), sqrt(2) * C1 * exp(sqrt(2) * x) / 2 + sqrt(2) * C3 * exp(-sqrt(2) * x) * Rational(-1, 2))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])
    eqs5 = [Eq(f(x).diff(x, 2), f(x)), Eq(g(x).diff(x, 2), f(x))]
    sol5 = [Eq(f(x), -C1 * exp(-x) + C2 * exp(x)), Eq(g(x), -C1 * exp(-x) + C2 * exp(x) + C3 + C4 * x)]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])
    eqs6 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x)), Eq(Derivative(g(x), (x, 2)), -f(x) - g(x))]
    sol6 = [Eq(f(x), C1 + C2 * x ** 2 / 2 + C2 + C4 * x ** 3 / 6 + x * (C3 + C4)), Eq(g(x), -C1 + C2 * x ** 2 * Rational(-1, 2) - C3 * x + C4 * x ** 3 * Rational(-1, 6))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])
    eqs7 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x) + 1), Eq(Derivative(g(x), (x, 2)), f(x) + g(x) + 1)]
    sol7 = [Eq(f(x), -C1 - C2 * x + sqrt(2) * C3 * exp(sqrt(2) * x) / 2 + sqrt(2) * C4 * exp(-sqrt(2) * x) * Rational(-1, 2) + Rational(-1, 2)), Eq(g(x), C1 + C2 * x + sqrt(2) * C3 * exp(sqrt(2) * x) / 2 + sqrt(2) * C4 * exp(-sqrt(2) * x) * Rational(-1, 2) + Rational(-1, 2))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])
    eqs8 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x) + 1), Eq(Derivative(g(x), (x, 2)), -f(x) - g(x) + 1)]
    sol8 = [Eq(f(x), C1 + C2 + C4 * x ** 3 / 6 + x ** 4 / 12 + x ** 2 * (C2 / 2 + Rational(1, 2)) + x * (C3 + C4)), Eq(g(x), -C1 - C3 * x + C4 * x ** 3 * Rational(-1, 6) + x ** 4 * Rational(-1, 12) - x ** 2 * (C2 / 2 + Rational(-1, 2)))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])
    x, y = symbols('x, y', cls=Function)
    t, l = symbols('t, l')
    eqs10 = [Eq(Derivative(x(t), (t, 2)), 5 * x(t) + 43 * y(t)), Eq(Derivative(y(t), (t, 2)), x(t) + 9 * y(t))]
    sol10 = [Eq(x(t), C1 * (61 - 9 * sqrt(47)) * sqrt(sqrt(47) + 7) * exp(-t * sqrt(sqrt(47) + 7)) / 2 + C2 * sqrt(7 - sqrt(47)) * (61 + 9 * sqrt(47)) * exp(-t * sqrt(7 - sqrt(47))) / 2 + C3 * (61 - 9 * sqrt(47)) * sqrt(sqrt(47) + 7) * exp(t * sqrt(sqrt(47) + 7)) * Rational(-1, 2) + C4 * sqrt(7 - sqrt(47)) * (61 + 9 * sqrt(47)) * exp(t * sqrt(7 - sqrt(47))) * Rational(-1, 2)), Eq(y(t), C1 * (7 - sqrt(47)) * sqrt(sqrt(47) + 7) * exp(-t * sqrt(sqrt(47) + 7)) * Rational(-1, 2) + C2 * sqrt(7 - sqrt(47)) * (sqrt(47) + 7) * exp(-t * sqrt(7 - sqrt(47))) * Rational(-1, 2) + C3 * (7 - sqrt(47)) * sqrt(sqrt(47) + 7) * exp(t * sqrt(sqrt(47) + 7)) / 2 + C4 * sqrt(7 - sqrt(47)) * (sqrt(47) + 7) * exp(t * sqrt(7 - sqrt(47))) / 2)]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])
    eqs11 = [Eq(7 * x(t) + Derivative(x(t), (t, 2)) - 9 * Derivative(y(t), t), 0), Eq(7 * y(t) + 9 * Derivative(x(t), t) + Derivative(y(t), (t, 2)), 0)]
    sol11 = [Eq(y(t), C1 * (9 - sqrt(109)) * sin(sqrt(2) * t * sqrt(9 * sqrt(109) + 95) / 2) / 14 + C2 * (9 - sqrt(109)) * cos(sqrt(2) * t * sqrt(9 * sqrt(109) + 95) / 2) * Rational(-1, 14) + C3 * (9 + sqrt(109)) * sin(sqrt(2) * t * sqrt(95 - 9 * sqrt(109)) / 2) / 14 + C4 * (9 + sqrt(109)) * cos(sqrt(2) * t * sqrt(95 - 9 * sqrt(109)) / 2) * Rational(-1, 14)), Eq(x(t), C1 * (9 - sqrt(109)) * cos(sqrt(2) * t * sqrt(9 * sqrt(109) + 95) / 2) * Rational(-1, 14) + C2 * (9 - sqrt(109)) * sin(sqrt(2) * t * sqrt(9 * sqrt(109) + 95) / 2) * Rational(-1, 14) + C3 * (9 + sqrt(109)) * cos(sqrt(2) * t * sqrt(95 - 9 * sqrt(109)) / 2) / 14 + C4 * (9 + sqrt(109)) * sin(sqrt(2) * t * sqrt(95 - 9 * sqrt(109)) / 2) / 14)]
    assert dsolve(eqs11) == sol11
    assert checksysodesol(eqs11, sol11) == (True, [0, 0])
    eqs13 = [Eq(Derivative(f(t), (t, 2)), Derivative(f(t), t) / t + f(t) / t ** 2 + g(t) / t ** 2), Eq(Derivative(g(t), (t, 2)), g(t) / t ** 2)]
    sol13 = [Eq(f(t), C1 * (sqrt(5) + 3) * Rational(-1, 2) * t ** (Rational(1, 2) + sqrt(5) * Rational(-1, 2)) + C2 * t ** (Rational(1, 2) + sqrt(5) / 2) * (3 - sqrt(5)) * Rational(-1, 2) - C3 * t ** (1 - sqrt(2)) * (1 + sqrt(2)) - C4 * t ** (1 + sqrt(2)) * (1 - sqrt(2))), Eq(g(t), C1 * (1 + sqrt(5)) * Rational(-1, 2) * t ** (Rational(1, 2) + sqrt(5) * Rational(-1, 2)) + C2 * t ** (Rational(1, 2) + sqrt(5) / 2) * (1 - sqrt(5)) * Rational(-1, 2))]
    assert dsolve(eqs13) == sol13
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])
    eqs14 = [Eq(Derivative(f(t), (t, 2)), t * f(t)), Eq(Derivative(g(t), (t, 2)), t * g(t))]
    sol14 = [Eq(f(t), C1 * airyai(t) + C2 * airybi(t)), Eq(g(t), C3 * airyai(t) + C4 * airybi(t))]
    assert dsolve(eqs14) == sol14
    assert checksysodesol(eqs14, sol14) == (True, [0, 0])
    eqs15 = [Eq(Derivative(x(t), (t, 2)), t * (4 * Derivative(x(t), t) + 8 * Derivative(y(t), t))), Eq(Derivative(y(t), (t, 2)), t * (12 * Derivative(x(t), t) - 6 * Derivative(y(t), t)))]
    sol15 = [Eq(x(t), C1 - erf(sqrt(6) * t) * (sqrt(6) * sqrt(pi) * C2 / 33 + sqrt(6) * sqrt(pi) * C3 * Rational(-1, 44)) + erfi(sqrt(5) * t) * (sqrt(5) * sqrt(pi) * C2 * Rational(2, 55) + sqrt(5) * sqrt(pi) * C3 * Rational(4, 55))), Eq(y(t), C4 + erf(sqrt(6) * t) * (sqrt(6) * sqrt(pi) * C2 * Rational(2, 33) + sqrt(6) * sqrt(pi) * C3 * Rational(-1, 22)) + erfi(sqrt(5) * t) * (sqrt(5) * sqrt(pi) * C2 * Rational(3, 110) + sqrt(5) * sqrt(pi) * C3 * Rational(3, 55)))]
    assert dsolve(eqs15) == sol15
    assert checksysodesol(eqs15, sol15) == (True, [0, 0])