from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
def test_match_riccati():
    """
    This function tests if an ODE is Riccati or not.

    Each test case has 5 values -

    1. eq - The Riccati ODE.
    2. match - Boolean indicating if eq is a Riccati ODE.
    3. b0 -
    4. b1 - Coefficient of f(x) in eq.
    5. b2 - Coefficient of f(x)**2 in eq.
    """
    tests = [(f(x).diff(x) - (405 * x ** 3 - 882 * x ** 2 - 78 * x + 92) / (243 * x ** 4 - 945 * x ** 3 + 846 * x ** 2 + 180 * x - 72) - 2 - f(x) ** 2 / (3 * x + 1) - (S(1) / 3 - x) * f(x) / (S(1) / 3 - 3 * x / 2), True, 45 * x ** 3 / (27 * x ** 4 - 105 * x ** 3 + 94 * x ** 2 + 20 * x - 8) - 98 * x ** 2 / (27 * x ** 4 - 105 * x ** 3 + 94 * x ** 2 + 20 * x - 8) - 26 * x / (81 * x ** 4 - 315 * x ** 3 + 282 * x ** 2 + 60 * x - 24) + 2 + 92 / (243 * x ** 4 - 945 * x ** 3 + 846 * x ** 2 + 180 * x - 72), Mul(-1, 2 - 6 * x, evaluate=False) / (9 * x - 2), 1 / (3 * x + 1)), (f(x).diff(x) + 4 * x / 27 - (x / 3 - 1) * f(x) ** 2 - (2 * x / 3 + 1) * f(x) / (3 * x + 2) - S(10) / 27 - (265 * x ** 2 + 423 * x + 162) / (324 * x ** 3 + 216 * x ** 2), True, -4 * x / 27 + S(10) / 27 + 3 / (6 * x ** 3 + 4 * x ** 2) + 47 / (36 * x ** 2 + 24 * x) + 265 / (324 * x + 216), Mul(-1, -2 * x - 3, evaluate=False) / (9 * x + 6), x / 3 - 1), (f(x).diff(x) - (304 * x ** 5 - 745 * x ** 4 + 631 * x ** 3 - 876 * x ** 2 + 198 * x - 108) / (36 * x ** 6 - 216 * x ** 5 + 477 * x ** 4 - 567 * x ** 3 + 360 * x ** 2 - 108 * x) - S(17) / 9 - (x - S(3) / 2) * f(x) / (x / 2 - S(3) / 2) - (x / 3 - 3) * f(x) ** 2 / (3 * x), True, 304 * x ** 4 / (36 * x ** 5 - 216 * x ** 4 + 477 * x ** 3 - 567 * x ** 2 + 360 * x - 108) - 745 * x ** 3 / (36 * x ** 5 - 216 * x ** 4 + 477 * x ** 3 - 567 * x ** 2 + 360 * x - 108) + 631 * x ** 2 / (36 * x ** 5 - 216 * x ** 4 + 477 * x ** 3 - 567 * x ** 2 + 360 * x - 108) - 292 * x / (12 * x ** 5 - 72 * x ** 4 + 159 * x ** 3 - 189 * x ** 2 + 120 * x - 36) + S(17) / 9 - 12 / (4 * x ** 6 - 24 * x ** 5 + 53 * x ** 4 - 63 * x ** 3 + 40 * x ** 2 - 12 * x) + 22 / (4 * x ** 5 - 24 * x ** 4 + 53 * x ** 3 - 63 * x ** 2 + 40 * x - 12), Mul(-1, 3 - 2 * x, evaluate=False) / (x - 3), Mul(-1, 9 - x, evaluate=False) / (9 * x)), (f(x).diff(x) - x ** (S(3) / 2) / (x ** (S(1) / 2) - 2) + x ** 2 * f(x) + x * f(x) ** 2 / x ** (S(3) / 4), False, 0, 0, 0), (f(x).diff(x) - sin(x ** 2) + exp(x) * f(x) + log(x) * f(x) ** 2, False, 0, 0, 0), (f(x).diff(x) - tanh(x + sqrt(x)) + f(x) + x ** 4 * f(x) ** 2, False, 0, 0, 0), ((1 - x ** 2) * f(x).diff(x, 2) - 2 * x * f(x).diff(x) + 20 * f(x), False, 0, 0, 0), (f(x).diff(x) - x ** 2 + x ** 3 * f(x) + x ** 2 / (x + 1) * f(x) ** 3, False, 0, 0, 0), (f(x).diff(x) * f(x) ** 2 + (x ** 2 - 1) / (x ** 3 + 1) * f(x) + 1 / (2 * x + 3) + f(x) ** 2, False, 0, 0, 0)]
    for eq, res, b0, b1, b2 in tests:
        match, funcs = match_riccati(eq, f, x)
        assert match == res
        if res:
            assert [b0, b1, b2] == funcs