from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify
from sympy.integrals.rationaltools import ratint, ratint_logpart, log_to_atan
from sympy.abc import a, b, x, t
def test_issues_8246_12050_13501_14080():
    a = symbols('a', nonzero=True)
    assert integrate(a / (x ** 2 + a ** 2), x) == atan(x / a)
    assert integrate(1 / (x ** 2 + a ** 2), x) == atan(x / a) / a
    assert integrate(1 / (1 + a ** 2 * x ** 2), x) == atan(a * x) / a