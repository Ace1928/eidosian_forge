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
def test_gruntz_eval_special_slow():
    _sskip()
    assert gruntz(gamma(x + 1) / sqrt(2 * pi) - exp(-x) * (x ** (x + S.Half) + x ** (x - S.Half) / 12), x, oo) is oo
    assert gruntz(exp(exp(exp(digamma(digamma(digamma(x)))))) / x, x, oo) == 0