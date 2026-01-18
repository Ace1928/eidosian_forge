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
def test_meromorphic():
    assert besselj(2, x).is_meromorphic(x, 1) == True
    assert besselj(2, x).is_meromorphic(x, 0) == True
    assert besselj(2, x).is_meromorphic(x, oo) == False
    assert besselj(S(2) / 3, x).is_meromorphic(x, 1) == True
    assert besselj(S(2) / 3, x).is_meromorphic(x, 0) == False
    assert besselj(S(2) / 3, x).is_meromorphic(x, oo) == False
    assert besselj(x, 2 * x).is_meromorphic(x, 2) == False
    assert besselk(0, x).is_meromorphic(x, 1) == True
    assert besselk(2, x).is_meromorphic(x, 0) == True
    assert besseli(0, x).is_meromorphic(x, 1) == True
    assert besseli(2, x).is_meromorphic(x, 0) == True
    assert bessely(0, x).is_meromorphic(x, 1) == True
    assert bessely(0, x).is_meromorphic(x, 0) == False
    assert bessely(2, x).is_meromorphic(x, 0) == True
    assert hankel1(3, x ** 2 + 2 * x).is_meromorphic(x, 1) == True
    assert hankel1(0, x).is_meromorphic(x, 0) == False
    assert hankel2(11, 4).is_meromorphic(x, 5) == True
    assert hn1(6, 7 * x ** 3 + 4).is_meromorphic(x, 7) == True
    assert hn2(3, 2 * x).is_meromorphic(x, 9) == True
    assert jn(5, 2 * x + 7).is_meromorphic(x, 4) == True
    assert yn(8, x ** 2 + 11).is_meromorphic(x, 6) == True