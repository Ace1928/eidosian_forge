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
def test_issue_7638():
    f = pi / log(sqrt(2))
    assert ((1 + I) ** (I * f / 2)) ** 0.3 == (1 + I) ** (0.15 * I * f)
    assert (1 + I) ** (4 * I * f) == ((1 + I) ** (12 * I * f)) ** Rational(1, 3)
    assert (((1 + I) ** (I * (1 + 7 * f))) ** Rational(1, 3)).exp == Rational(1, 3)
    r = symbols('r', real=True)
    assert sqrt(r ** 2) == abs(r)
    assert cbrt(r ** 3) != r
    assert sqrt(Pow(2 * I, 5 * S.Half)) != (2 * I) ** Rational(5, 4)
    p = symbols('p', positive=True)
    assert cbrt(p ** 2) == p ** Rational(2, 3)
    assert NS(((0.2 + 0.7 * I) ** (0.7 + 1.0 * I)) ** (0.5 - 0.1 * I), 1) == '0.4 + 0.2*I'
    assert sqrt(1 / (1 + I)) == sqrt(1 - I) / sqrt(2)
    e = 1 / (1 - sqrt(2))
    assert sqrt(e) == I / sqrt(-1 + sqrt(2))
    assert e ** Rational(-1, 2) == -I * sqrt(-1 + sqrt(2))
    assert sqrt((cos(1) ** 2 + sin(1) ** 2 - 1) ** (3 + I)).exp in [S.Half, Rational(3, 2) + I / 2]
    assert sqrt(r ** Rational(4, 3)) != r ** Rational(2, 3)
    assert sqrt((p + I) ** Rational(4, 3)) == (p + I) ** Rational(2, 3)
    for q in (1 + I, 1 - I):
        assert sqrt(q ** 2) == q
    for q in (-1 + I, -1 - I):
        assert sqrt(q ** 2) == -q
    assert sqrt((p + r * I) ** 2) != p + r * I
    e = 1 + I / 5
    assert sqrt(e ** 5) == e ** (5 * S.Half)
    assert sqrt(e ** 6) == e ** 3
    assert sqrt((1 + I * r) ** 6) != (1 + I * r) ** 3