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
def test_construct_d_case_6():
    """
    This function tests the Case 6 in the step
    to calculate coefficients of the d-vector.

    Each test case has 3 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. d - The d-vector.
    """
    tests = [(Poly(-2 * x ** 2 - 5, x, domain='ZZ'), Poly(4 * x ** 4 + 2 * x ** 2 + 10 * x + 2, x, domain='ZZ'), [[S(1) / 2 + I / 2], [S(1) / 2 - I / 2]]), (Poly(-2 * x ** 3 - 4 * x ** 2 - 2 * x - 5, x, domain='ZZ'), Poly(x ** 6 - x ** 5 + 2 * x ** 4 - 4 * x ** 3 - 5 * x ** 2 - 5 * x + 9, x, domain='ZZ'), [[1], [0]]), (Poly(-5 * x ** 3 + x ** 2 + 11 * x + 12, x, domain='ZZ'), Poly(6 * x ** 8 - 26 * x ** 7 - 27 * x ** 6 - 10 * x ** 5 - 44 * x ** 4 - 46 * x ** 3 - 34 * x ** 2 - 27 * x - 42, x, domain='ZZ'), [[1], [0]])]
    for num, den, d in tests:
        assert construct_d_case_6(num, den, x) == d