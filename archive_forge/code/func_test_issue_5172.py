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
@XFAIL
def test_issue_5172():
    n = Symbol('n')
    r = Symbol('r', positive=True)
    c = Symbol('c')
    p = Symbol('p', positive=True)
    m = Symbol('m', negative=True)
    expr = ((2 * n * (n - r + 1) / (n + r * (n - r + 1))) ** c + (r - 1) * (n * (n - r + 2) / (n + r * (n - r + 1))) ** c - n) / (n ** c - n)
    expr = expr.subs(c, c + 1)
    assert gruntz(expr.subs(c, m), n, oo) == 1
    assert gruntz(expr.subs(c, p), n, oo).simplify() == (2 ** (p + 1) + r - 1) / (r + 1) ** (p + 1)