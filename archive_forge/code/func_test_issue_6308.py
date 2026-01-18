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
def test_issue_6308():
    k, a0 = symbols('k a0', real=True)
    assert integrate((x ** 2 + 1 - k ** 2) / (x ** 2 + 1 + a0 ** 2), x) == x - (a0 ** 2 + k ** 2) * atan(x / sqrt(a0 ** 2 + 1)) / sqrt(a0 ** 2 + 1)