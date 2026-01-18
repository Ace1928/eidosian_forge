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
def test_erf2inv():
    assert erf2inv(0, 0) is S.Zero
    assert erf2inv(0, 1) is S.Infinity
    assert erf2inv(1, 0) is S.One
    assert erf2inv(0, y) == erfinv(y)
    assert erf2inv(oo, y) == erfcinv(-y)
    assert erf2inv(x, 0) == x
    assert erf2inv(x, oo) == erfinv(x)
    assert erf2inv(nan, 0) is nan
    assert erf2inv(0, nan) is nan
    assert erf2inv(x, y).diff(x) == exp(-x ** 2 + erf2inv(x, y) ** 2)
    assert erf2inv(x, y).diff(y) == sqrt(pi) * exp(erf2inv(x, y) ** 2) / 2
    raises(ArgumentIndexError, lambda: erf2inv(x, y).fdiff(3))