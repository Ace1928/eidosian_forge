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
def test_construct_d_case_4():
    """
    This function tests the Case 4 in the step
    to calculate coefficients of the d-vector.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. mul - Multiplicity of oo as a pole.
    4. d - The d-vector.
    """
    tests = [(Poly(-x ** 5 - 2 * x ** 4 + 4 * x ** 3 + 2 * x + 5, x, extension=True), Poly(9 * x ** 3 - 2 * x ** 2 + 10 * x - 2, x, extension=True), 2, [[10 * I / 27, I / 3, -3 * I * (S(158) / 243 - I / 3) / 2], [-10 * I / 27, -I / 3, 3 * I * (S(158) / 243 + I / 3) / 2]]), (Poly(-x ** 6 + 9 * x ** 5 + 5 * x ** 4 + 6 * x ** 3 + 5 * x ** 2 + 6 * x + 7, x, extension=True), Poly(x ** 4 + 3 * x ** 3 + 12 * x ** 2 - x + 7, x, extension=True), 2, [[-6 * I, I, -I * (17 - I) / 2], [6 * I, -I, I * (17 + I) / 2]]), (Poly(-2 * x ** 6 - x ** 5 - x ** 4 - 2 * x ** 3 - x ** 2 - 3 * x - 3, x, extension=True), Poly(3 * x ** 2 + 10 * x + 7, x, extension=True), 4, [[269 * sqrt(6) * I / 288, -17 * sqrt(6) * I / 36, sqrt(6) * I / 3, -sqrt(6) * I * (S(16969) / 2592 - 2 * sqrt(6) * I / 3) / 4], [-269 * sqrt(6) * I / 288, 17 * sqrt(6) * I / 36, -sqrt(6) * I / 3, sqrt(6) * I * (S(16969) / 2592 + 2 * sqrt(6) * I / 3) / 4]]), (Poly(-3 * x ** 5 - 3 * x ** 4 - 3 * x ** 3 - x ** 2 - 1, x, extension=True), Poly(12 * x - 2, x, extension=True), 4, [[41 * I / 192, 7 * I / 24, I / 2, -I * (-S(59) / 6912 - I)], [-41 * I / 192, -7 * I / 24, -I / 2, I * (-S(59) / 6912 + I)]]), (Poly(-x ** 7 - x ** 5 - x ** 4 - x ** 2 - x, x, extension=True), Poly(x + 2, x, extension=True), 6, [[-5 * I / 2, 2 * I, -I, I, -I * (-9 - 3 * I) / 2], [5 * I / 2, -2 * I, I, -I, I * (-9 + 3 * I) / 2]]), (Poly(-x ** 7 - x ** 6 - 2 * x ** 5 - 2 * x ** 4 - x ** 3 - x ** 2 + 2 * x - 2, x, extension=True), Poly(2 * x - 2, x, extension=True), 6, [[3 * sqrt(2) * I / 4, 3 * sqrt(2) * I / 4, sqrt(2) * I / 2, sqrt(2) * I / 2, -sqrt(2) * I * (-S(7) / 8 - 3 * sqrt(2) * I / 2) / 2], [-3 * sqrt(2) * I / 4, -3 * sqrt(2) * I / 4, -sqrt(2) * I / 2, -sqrt(2) * I / 2, sqrt(2) * I * (-S(7) / 8 + 3 * sqrt(2) * I / 2) / 2]])]
    for num, den, mul, d in tests:
        ser = rational_laurent_series(num, den, x, oo, mul, 1)
        assert construct_d_case_4(ser, mul // 2) == d