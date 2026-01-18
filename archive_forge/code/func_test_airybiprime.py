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
def test_airybiprime():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)
    assert isinstance(airybiprime(z), airybiprime)
    assert airybiprime(0) == 3 ** Rational(1, 6) / gamma(Rational(1, 3))
    assert airybiprime(oo) is oo
    assert airybiprime(-oo) == 0
    assert diff(airybiprime(z), z) == z * airybi(z)
    assert series(airybiprime(z), z, 0, 3) == 3 ** Rational(1, 6) / gamma(Rational(1, 3)) + 3 ** Rational(5, 6) * z ** 2 / (6 * gamma(Rational(2, 3))) + O(z ** 3)
    assert airybiprime(z).rewrite(hyper) == 3 ** Rational(5, 6) * z ** 2 * hyper((), (Rational(5, 3),), z ** 3 / 9) / (6 * gamma(Rational(2, 3))) + 3 ** Rational(1, 6) * hyper((), (Rational(1, 3),), z ** 3 / 9) / gamma(Rational(1, 3))
    assert isinstance(airybiprime(z).rewrite(besselj), airybiprime)
    assert airyai(t).rewrite(besselj) == sqrt(-t) * (besselj(Rational(-1, 3), 2 * (-t) ** Rational(3, 2) / 3) + besselj(Rational(1, 3), 2 * (-t) ** Rational(3, 2) / 3)) / 3
    assert airybiprime(z).rewrite(besseli) == sqrt(3) * (z ** 2 * besseli(Rational(2, 3), 2 * z ** Rational(3, 2) / 3) / (z ** Rational(3, 2)) ** Rational(2, 3) + (z ** Rational(3, 2)) ** Rational(2, 3) * besseli(Rational(-2, 3), 2 * z ** Rational(3, 2) / 3)) / 3
    assert airybiprime(p).rewrite(besseli) == sqrt(3) * p * (besseli(Rational(-2, 3), 2 * p ** Rational(3, 2) / 3) + besseli(Rational(2, 3), 2 * p ** Rational(3, 2) / 3)) / 3
    assert expand_func(airybiprime(2 * (3 * z ** 5) ** Rational(1, 3))) == sqrt(3) * (z ** Rational(5, 3) / (z ** 5) ** Rational(1, 3) - 1) * airyaiprime(2 * 3 ** Rational(1, 3) * z ** Rational(5, 3)) / 2 + (z ** Rational(5, 3) / (z ** 5) ** Rational(1, 3) + 1) * airybiprime(2 * 3 ** Rational(1, 3) * z ** Rational(5, 3)) / 2