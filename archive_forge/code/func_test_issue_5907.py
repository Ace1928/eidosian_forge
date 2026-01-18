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
def test_issue_5907():
    a = symbols('a', nonzero=True)
    assert integrate(1 / (x ** 2 + a ** 2) ** 2, x) == x / (2 * a ** 4 + 2 * a ** 2 * x ** 2) + atan(x / a) / (2 * a ** 3)