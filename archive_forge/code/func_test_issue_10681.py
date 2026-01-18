from sympy.core.function import expand_func
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import cosh, acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
from sympy.functions.special.error_functions import (erf, erfc)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
from sympy.testing.pytest import slow
from sympy.core.random import (verify_numerically,
from sympy.abc import x, y, a, b, c, d, s, t, z
def test_issue_10681():
    from sympy.polys.domains.realfield import RR
    from sympy.abc import R, r
    f = integrate(r ** 2 * (R ** 2 - r ** 2) ** 0.5, r, meijerg=True)
    g = 1.0 / 3 * R ** 1.0 * r ** 3 * hyper((-0.5, Rational(3, 2)), (Rational(5, 2),), r ** 2 * exp_polar(2 * I * pi) / R ** 2)
    assert RR.almosteq((f / g).n(), 1.0, 1e-12)