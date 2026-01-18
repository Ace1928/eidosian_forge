from sympy.core.function import (diff, expand, expand_func)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Chi, Ci, E1, Ei, Li, Shi, Si, erf, erf2, erf2inv, erfc, erfcinv, erfi, erfinv, expint, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.gruntz import gruntz
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.special.error_functions import _erfs, _eis
from sympy.testing.pytest import raises
def test__eis():
    assert _eis(z).diff(z) == -_eis(z) + 1 / z
    assert _eis(1 / z).series(z) == z + z ** 2 + 2 * z ** 3 + 6 * z ** 4 + 24 * z ** 5 + O(z ** 6)
    assert Ei(z).rewrite('tractable') == exp(z) * _eis(z)
    assert li(z).rewrite('tractable') == z * _eis(log(z))
    assert _eis(z).rewrite('intractable') == exp(-z) * Ei(z)
    assert expand(li(z).rewrite('tractable').diff(z).rewrite('intractable')) == li(z).diff(z)
    assert expand(Ei(z).rewrite('tractable').diff(z).rewrite('intractable')) == Ei(z).diff(z)
    assert _eis(z).series(z, n=3) == EulerGamma + log(z) + z * (-log(z) - EulerGamma + 1) + z ** 2 * (log(z) / 2 - Rational(3, 4) + EulerGamma / 2) + O(z ** 3 * log(z))
    raises(ArgumentIndexError, lambda: _eis(z).fdiff(2))