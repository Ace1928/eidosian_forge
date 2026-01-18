from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acot, atan, cos, sin)
from sympy.functions.elementary.complexes import sign as _sign
from sympy.functions.special.error_functions import (Ei, erf)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.polys.polytools import cancel
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.series.gruntz import compare, mrv, rewrite, mrv_leadterm, gruntz, \
from sympy.testing.pytest import XFAIL, skip, slow
def test_compare1():
    assert compare(2, x, x) == '<'
    assert compare(x, exp(x), x) == '<'
    assert compare(exp(x), exp(x ** 2), x) == '<'
    assert compare(exp(x ** 2), exp(exp(x)), x) == '<'
    assert compare(1, exp(exp(x)), x) == '<'
    assert compare(x, 2, x) == '>'
    assert compare(exp(x), x, x) == '>'
    assert compare(exp(x ** 2), exp(x), x) == '>'
    assert compare(exp(exp(x)), exp(x ** 2), x) == '>'
    assert compare(exp(exp(x)), 1, x) == '>'
    assert compare(2, 3, x) == '='
    assert compare(3, -5, x) == '='
    assert compare(2, -5, x) == '='
    assert compare(x, x ** 2, x) == '='
    assert compare(x ** 2, x ** 3, x) == '='
    assert compare(x ** 3, 1 / x, x) == '='
    assert compare(1 / x, x ** m, x) == '='
    assert compare(x ** m, -x, x) == '='
    assert compare(exp(x), exp(-x), x) == '='
    assert compare(exp(-x), exp(2 * x), x) == '='
    assert compare(exp(2 * x), exp(x) ** 2, x) == '='
    assert compare(exp(x) ** 2, exp(x + exp(-x)), x) == '='
    assert compare(exp(x), exp(x + exp(-x)), x) == '='
    assert compare(exp(x ** 2), 1 / exp(x ** 2), x) == '='