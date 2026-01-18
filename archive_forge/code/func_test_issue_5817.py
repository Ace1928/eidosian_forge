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
def test_issue_5817():
    a, b, c = symbols('a,b,c', positive=True)
    assert simplify(ratint(a / (b * c * x ** 2 + a ** 2 + b * a), x)) == sqrt(a) * atan(sqrt(b) * sqrt(c) * x / (sqrt(a) * sqrt(a + b))) / (sqrt(b) * sqrt(c) * sqrt(a + b))