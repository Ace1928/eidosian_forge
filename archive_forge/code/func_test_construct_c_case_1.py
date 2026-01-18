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
def test_construct_c_case_1():
    """
    This function tests the Case 1 in the step
    to calculate coefficients of c-vectors.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. c - The c-vector for the pole.
    """
    tests = [(Poly(-3 * x ** 3 + 3 * x ** 2 + 4 * x - 5, x, extension=True), Poly(4 * x ** 8 + 16 * x ** 7 + 9 * x ** 5 + 12 * x ** 4 + 6 * x ** 3 + 12 * x ** 2, x, extension=True), S(0), [[S(1) / 2 + sqrt(6) * I / 6], [S(1) / 2 - sqrt(6) * I / 6]]), (Poly(1200 * x ** 3 + 1440 * x ** 2 + 816 * x + 560, x, extension=True), Poly(128 * x ** 5 - 656 * x ** 4 + 1264 * x ** 3 - 1125 * x ** 2 + 385 * x + 49, x, extension=True), S(7) / 4, [[S(1) / 2 + sqrt(16367978) / 634], [S(1) / 2 - sqrt(16367978) / 634]]), (Poly(4 * x + 2, x, extension=True), Poly(18 * x ** 4 + (2 - 18 * sqrt(3)) * x ** 3 + (14 - 11 * sqrt(3)) * x ** 2 + (4 - 6 * sqrt(3)) * x + 8 * sqrt(3) + 16, x, domain='QQ<sqrt(3)>'), (S(1) + sqrt(3)) / 2, [[S(1) / 2 + sqrt(Mul(4, 2 * sqrt(3) + 4, evaluate=False) / (19 * sqrt(3) + 44) + 1) / 2], [S(1) / 2 - sqrt(Mul(4, 2 * sqrt(3) + 4, evaluate=False) / (19 * sqrt(3) + 44) + 1) / 2]])]
    for num, den, pole, c in tests:
        assert construct_c_case_1(num, den, x, pole) == c