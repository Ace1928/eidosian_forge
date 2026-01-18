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
def test_issue_21647():
    e = log((Integer(567) / 500) ** (811 * (Integer(567) / 500) ** x / 100))
    ans = log(Mul(Rational(64701150190720499096094005280169087619821081527, 76293945312500000000000000000000000000000000000), Pow(Integer(2), Rational(396204892125479941, 781250000000000000)), Pow(Integer(3), Rational(385045107874520059, 390625000000000000)), Pow(Integer(5), Rational(407364676376439823, 1562500000000000000)), Pow(Integer(7), Rational(385045107874520059, 1562500000000000000))))
    assert e.xreplace({x: 6}) == ans