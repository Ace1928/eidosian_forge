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
def test_besseli_leading_term():
    assert besseli(0, x).as_leading_term(x) == 1
    assert besseli(1, sin(x)).as_leading_term(x) == x / 2
    assert besseli(1, 2 * sqrt(x)).as_leading_term(x) == sqrt(x)