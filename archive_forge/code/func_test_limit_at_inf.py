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
def test_limit_at_inf():
    """
    This function tests the limit at oo of a
    rational function.

    Each test case has 3 values -

    1. num - Numerator of rational function.
    2. den - Denominator of rational function.
    3. limit_at_inf - Limit of rational function at oo
    """
    tests = [(Poly(-12 * x ** 2 + 20 * x + 32, x), Poly(32 * x ** 3 + 72 * x ** 2 + 3 * x - 32, x), 0), (Poly(1260 * x ** 4 - 1260 * x ** 3 - 700 * x ** 2 - 1260 * x + 1400, x), Poly(6300 * x ** 3 - 1575 * x ** 2 + 756 * x - 540, x), oo), (Poly(-735 * x ** 8 - 1400 * x ** 7 + 1680 * x ** 6 - 315 * x ** 5 - 600 * x ** 4 + 840 * x ** 3 - 525 * x ** 2 + 630 * x + 3780, x), Poly(1008 * x ** 7 - 2940 * x ** 6 - 84 * x ** 5 + 2940 * x ** 4 - 420 * x ** 3 + 1512 * x ** 2 + 105 * x + 168, x), -oo), (Poly(105 * x ** 7 - 960 * x ** 6 + 60 * x ** 5 + 60 * x ** 4 - 80 * x ** 3 + 45 * x ** 2 + 120 * x + 15, x), Poly(735 * x ** 7 + 525 * x ** 6 + 720 * x ** 5 + 720 * x ** 4 - 8400 * x ** 3 - 2520 * x ** 2 + 2800 * x + 280, x), S(1) / 7), (Poly(288 * x ** 4 - 450 * x ** 3 + 280 * x ** 2 - 900 * x - 90, x), Poly(607 * x ** 4 + 840 * x ** 3 - 1050 * x ** 2 + 420 * x + 420, x), S(288) / 607)]
    for num, den, lim in tests:
        assert limit_at_inf(num, den, x) == lim