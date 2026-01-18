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
def test_issue_6100_12942_4473():
    assert x ** 1.0 != x
    assert x != x ** 1.0
    assert True != x ** 1.0
    assert x ** 1.0 is not True
    assert x is not True
    assert x * y != (x * y) ** 1.0
    assert (x ** 1.0) ** 1.0 != x
    assert (x ** 1.0) ** 2.0 != x ** 2
    b = Expr()
    assert Pow(b, 1.0, evaluate=False) != b
    assert ((x * y) ** 1.0).func is Pow