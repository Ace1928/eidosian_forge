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
def test_limit3():
    a = Symbol('a')
    assert gruntz(x - log(1 + exp(x)), x, oo) == 0
    assert gruntz(x - log(a + exp(x)), x, oo) == 0
    assert gruntz(exp(x) / (1 + exp(x)), x, oo) == 1
    assert gruntz(exp(x) / (a + exp(x)), x, oo) == 1