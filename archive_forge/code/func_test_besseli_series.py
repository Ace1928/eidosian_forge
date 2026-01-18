from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, hankel1, hankel2, hn1, hn2, jn, jn_zeros, yn)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.series.series import series
from sympy.functions.special.bessel import (airyai, airybi,
from sympy.core.random import (random_complex_number as randcplx,
from sympy.simplify import besselsimp
from sympy.testing.pytest import raises, slow
from sympy.abc import z, n, k, x
def test_besseli_series():
    assert besseli(0, x).series(x) == 1 + x ** 2 / 4 + x ** 4 / 64 + O(x ** 6)
    assert besseli(0, x ** 1.1).series(x) == 1 + x ** 4.4 / 64 + x ** 2.2 / 4 + O(x ** 6)
    assert besseli(0, x ** 2 + x).series(x) == 1 + x ** 2 / 4 + x ** 3 / 2 + 17 * x ** 4 / 64 + x ** 5 / 16 + O(x ** 6)
    assert besseli(0, sqrt(x) + x).series(x, n=4) == 1 + x / 4 + 17 * x ** 2 / 64 + 217 * x ** 3 / 2304 + x ** (S(3) / 2) / 2 + x ** (S(5) / 2) / 16 + 25 * x ** (S(7) / 2) / 384 + O(x ** 4)
    assert besseli(0, x / (1 - x)).series(x) == 1 + x ** 2 / 4 + x ** 3 / 2 + 49 * x ** 4 / 64 + 17 * x ** 5 / 16 + O(x ** 6)
    assert besseli(0, log(1 + x)).series(x) == 1 + x ** 2 / 4 - x ** 3 / 4 + 47 * x ** 4 / 192 - 23 * x ** 5 / 96 + O(x ** 6)
    assert besseli(1, sin(x)).series(x) == x / 2 - x ** 3 / 48 - 47 * x ** 5 / 1920 + O(x ** 6)
    assert besseli(1, 2 * sqrt(x)).series(x) == sqrt(x) + x ** (S(3) / 2) / 2 + x ** (S(5) / 2) / 12 + x ** (S(7) / 2) / 144 + x ** (S(9) / 2) / 2880 + x ** (S(11) / 2) / 86400 + O(x ** 6)
    assert besseli(-2, sin(x)).series(x, n=4) == besseli(2, sin(x)).series(x, n=4)