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
def test_construct_c_case_2():
    """
    This function tests the Case 2 in the step
    to calculate coefficients of c-vectors.

    Each test case has 5 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. mul - The multiplicity of the pole.
    5. c - The c-vector for the pole.
    """
    tests = [(Poly(1, x, extension=True), Poly((x - 1) ** 2 * (x - 2), x, extension=True), 1, 2, [[-I * (-1 - I) / 2], [I * (-1 + I) / 2]]), (Poly(3 * x ** 5 - 12 * x ** 4 - 7 * x ** 3 + 1, x, extension=True), Poly((3 * x - 1) ** 2 * (x + 2) ** 2, x, extension=True), S(1) / 3, 2, [[-S(89) / 98], [-S(9) / 98]]), (Poly(x ** 3 - x ** 2 + 4 * x, x, extension=True), Poly((x - 2) ** 4 * (x + 5) ** 2, x, extension=True), 2, 4, [[7 * sqrt(3) * (S(60) / 343 - 4 * sqrt(3) / 7) / 12, 2 * sqrt(3) / 7], [-7 * sqrt(3) * (S(60) / 343 + 4 * sqrt(3) / 7) / 12, -2 * sqrt(3) / 7]]), (Poly(3 * x ** 5 + x ** 4 + 3, x, extension=True), Poly((4 * x + 1) ** 4 * (x + 2), x, extension=True), -S(1) / 4, 4, [[128 * sqrt(439) * (-sqrt(439) / 128 - S(55) / 14336) / 439, sqrt(439) / 256], [-128 * sqrt(439) * (sqrt(439) / 128 - S(55) / 14336) / 439, -sqrt(439) / 256]]), (Poly(x ** 3 + 2, x, extension=True), Poly((3 * x - 1) ** 6 * (x ** 2 + 1), x, extension=True), S(1) / 3, 6, [[27 * sqrt(66) * (-sqrt(66) / 54 - S(131) / 267300) / 22, -2 * sqrt(66) / 1485, sqrt(66) / 162], [-27 * sqrt(66) * (sqrt(66) / 54 - S(131) / 267300) / 22, 2 * sqrt(66) / 1485, -sqrt(66) / 162]]), (Poly(x ** 2 + 12, x, extension=True), Poly((x - sqrt(2)) ** 6, x, extension=True), sqrt(2), 6, [[sqrt(14) * (S(6) / 7 - 3 * sqrt(14)) / 28, sqrt(7) / 7, sqrt(14)], [-sqrt(14) * (S(6) / 7 + 3 * sqrt(14)) / 28, -sqrt(7) / 7, -sqrt(14)]])]
    for num, den, pole, mul, c in tests:
        assert construct_c_case_2(num, den, x, pole, mul) == c