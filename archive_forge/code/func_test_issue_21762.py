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
def test_issue_21762():
    e = (x ** 2 + 6) ** (Integer(33333333333333333) / 50000000000000000)
    ans = Mul(Rational(5, 4), Pow(Integer(2), Rational(16666666666666667, 25000000000000000)), Pow(Integer(5), Rational(8333333333333333, 25000000000000000)))
    assert e.xreplace({x: S.Half}) == ans