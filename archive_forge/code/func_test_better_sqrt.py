from sympy.core import (
from sympy.core.parameters import global_parameters
from sympy.core.tests.test_evalf import NS
from sympy.core.function import expand_multinomial
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.error_functions import erf
from sympy.functions.elementary.trigonometric import (
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.polys import Poly
from sympy.series.order import O
from sympy.sets import FiniteSet
from sympy.core.power import power, integer_nthroot
from sympy.testing.pytest import warns, _both_exp_pow
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.abc import a, b, c, x, y
def test_better_sqrt():
    n = Symbol('n', integer=True, nonnegative=True)
    assert sqrt(3 + 4 * I) == 2 + I
    assert sqrt(3 - 4 * I) == 2 - I
    assert sqrt(-3 - 4 * I) == 1 - 2 * I
    assert sqrt(-3 + 4 * I) == 1 + 2 * I
    assert sqrt(32 + 24 * I) == 6 + 2 * I
    assert sqrt(32 - 24 * I) == 6 - 2 * I
    assert sqrt(-32 - 24 * I) == 2 - 6 * I
    assert sqrt(-32 + 24 * I) == 2 + 6 * I
    assert sqrt((3 + 4 * I) / 4) == 1 + I / 2
    assert sqrt((8 + 15 * I) / 8) == (5 + 3 * I) / 4
    assert sqrt((3 - 4 * I) / 25) == (2 - I) / 5
    assert sqrt((3 - 4 * I) / 26) == (2 - I) / sqrt(26)
    assert sqrt((3 + 4 * I) / (3 - 4 * I)) == (3 + 4 * I) / 5
    assert sqrt(2 / (3 + 4 * I)) == sqrt(2) / 5 * (2 - I)
    assert sqrt(n / (3 + 4 * I)).subs(n, 2) == sqrt(2) / 5 * (2 - I)
    assert sqrt(-2 / (3 + 4 * I)) == sqrt(2) / 5 * (1 + 2 * I)
    assert sqrt(-n / (3 + 4 * I)).subs(n, 2) == sqrt(2) / 5 * (1 + 2 * I)
    assert sqrt(1 / (3 + I * 4)) == (2 - I) / 5
    assert sqrt(1 / (3 - I)) == sqrt(10) * sqrt(3 + I) / 10
    i = symbols('i', imaginary=True)
    assert sqrt(3 / i) == Mul(sqrt(3), 1 / sqrt(i), evaluate=False)
    assert sqrt(3 + 4 * I) ** 3 == (2 + I) ** 3
    assert Pow(3 + 4 * I, Rational(3, 2)) == 2 + 11 * I
    assert Pow(6 + 8 * I, Rational(3, 2)) == 2 * sqrt(2) * (2 + 11 * I)
    n, d = (3 + 4 * I, (3 - 4 * I) ** 3)
    a = n / d
    assert a.args == (1 / d, n)
    eq = sqrt(a)
    assert eq.args == (a, S.Half)
    assert expand_multinomial(eq) == sqrt((-117 + 44 * I) * (3 + 4 * I)) / 125
    assert eq.expand() == (7 - 24 * I) / 125
    assert sqrt(2 * I) == 1 + I
    assert sqrt(2 * 9 * I) == Mul(3, 1 + I, evaluate=False)
    assert Pow(2 * I, 3 * S.Half) == (1 + I) ** 3
    assert sqrt(-I / 2) == Mul(S.Half, 1 - I, evaluate=False)
    assert Pow(Rational(-9, 2) * I, Rational(3, 2)) == 27 * (1 - I) ** 3 / 8