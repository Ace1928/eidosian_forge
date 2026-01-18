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
def test_bessel():
    from sympy.functions.special.bessel import besseli, besselj
    assert simplify(integrate(besselj(a, z) * besselj(b, z) / z, (z, 0, oo), meijerg=True, conds='none')) == 2 * sin(pi * (a / 2 - b / 2)) / (pi * (a - b) * (a + b))
    assert simplify(integrate(besselj(a, z) * besselj(a, z) / z, (z, 0, oo), meijerg=True, conds='none')) == 1 / (2 * a)
    assert simplify(integrate(sin(z * x) * (x ** 2 - 1) ** (-(y + S.Half)), (x, 1, oo), meijerg=True, conds='none') * 2 / ((z / 2) ** y * sqrt(pi) * gamma(S.Half - y))) == besselj(y, z)
    assert integrate(x * besselj(0, x), x, meijerg=True) == x * besselj(1, x)
    assert integrate(x * besseli(0, x), x, meijerg=True) == x * besseli(1, x)
    assert integrate(besselj(1, x), x, meijerg=True) == -besselj(0, x)
    assert integrate(besselj(1, x) ** 2 / x, x, meijerg=True) == -(besselj(0, x) ** 2 + besselj(1, x) ** 2) / 2
    assert integrate(besselj(0, x) ** 2 / x ** 2, x, meijerg=True) == -2 * x * besselj(0, x) ** 2 - 2 * x * besselj(1, x) ** 2 + 2 * besselj(0, x) * besselj(1, x) - besselj(0, x) ** 2 / x
    assert integrate(besselj(0, x) * besselj(1, x), x, meijerg=True) == -besselj(0, x) ** 2 / 2
    assert integrate(x ** 2 * besselj(0, x) * besselj(1, x), x, meijerg=True) == x ** 2 * besselj(1, x) ** 2 / 2
    assert integrate(besselj(0, x) * besselj(1, x) / x, x, meijerg=True) == x * besselj(0, x) ** 2 + x * besselj(1, x) ** 2 - besselj(0, x) * besselj(1, x)
    assert integrate(besselj(1, x ** 2) * x, x, meijerg=True) == -besselj(0, x ** 2) / 2