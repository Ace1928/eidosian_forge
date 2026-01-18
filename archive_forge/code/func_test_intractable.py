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
def test_intractable():
    assert gruntz(1 / gamma(x), x, oo) == 0
    assert gruntz(1 / loggamma(x), x, oo) == 0
    assert gruntz(gamma(x) / loggamma(x), x, oo) is oo
    assert gruntz(exp(gamma(x)) / gamma(x), x, oo) is oo
    assert gruntz(gamma(x), x, 3) == 2
    assert gruntz(gamma(Rational(1, 7) + 1 / x), x, oo) == gamma(Rational(1, 7))
    assert gruntz(log(x ** x) / log(gamma(x)), x, oo) == 1
    assert gruntz(log(gamma(gamma(x))) / exp(x), x, oo) is oo