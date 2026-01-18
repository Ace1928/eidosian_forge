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
def test_fresnel_series():
    assert fresnelc(z).series(z, n=15) == z - pi ** 2 * z ** 5 / 40 + pi ** 4 * z ** 9 / 3456 - pi ** 6 * z ** 13 / 599040 + O(z ** 15)
    fs = S.Half - sin(pi * z ** 2 / 2) / (pi ** 2 * z ** 3) + (-1 / (pi * z) + 3 / (pi ** 3 * z ** 5)) * cos(pi * z ** 2 / 2)
    fc = S.Half - cos(pi * z ** 2 / 2) / (pi ** 2 * z ** 3) + (1 / (pi * z) - 3 / (pi ** 3 * z ** 5)) * sin(pi * z ** 2 / 2)
    assert fresnels(z).series(z, oo) == fs + O(z ** (-6), (z, oo))
    assert fresnelc(z).series(z, oo) == fc + O(z ** (-6), (z, oo))
    assert (fresnels(z).series(z, -oo) + fs.subs(z, -z)).expand().is_Order
    assert (fresnelc(z).series(z, -oo) + fc.subs(z, -z)).expand().is_Order
    assert (fresnels(1 / z).series(z) - fs.subs(z, 1 / z)).expand().is_Order
    assert (fresnelc(1 / z).series(z) - fc.subs(z, 1 / z)).expand().is_Order
    assert ((2 * fresnels(3 * z)).series(z, oo) - 2 * fs.subs(z, 3 * z)).expand().is_Order
    assert ((3 * fresnelc(2 * z)).series(z, oo) - 3 * fc.subs(z, 2 * z)).expand().is_Order