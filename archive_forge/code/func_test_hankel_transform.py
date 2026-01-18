from sympy.integrals.transforms import (
from sympy.integrals.laplace import (
from sympy.core.function import Function, expand_mul
from sympy.core import EulerGamma
from sympy.core.numbers import I, Rational, oo, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan, cos, sin, tan
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import erf, expint
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.trigsimp import trigsimp
from sympy.testing.pytest import XFAIL, slow, skip, raises
from sympy.abc import x, s, a, b, c, d
def test_hankel_transform():
    r = Symbol('r')
    k = Symbol('k')
    nu = Symbol('nu')
    m = Symbol('m')
    a = symbols('a')
    assert hankel_transform(1 / r, r, k, 0) == 1 / k
    assert inverse_hankel_transform(1 / k, k, r, 0) == 1 / r
    assert hankel_transform(1 / r ** m, r, k, 0) == 2 ** (-m + 1) * k ** (m - 2) * gamma(-m / 2 + 1) / gamma(m / 2)
    assert inverse_hankel_transform(2 ** (-m + 1) * k ** (m - 2) * gamma(-m / 2 + 1) / gamma(m / 2), k, r, 0) == r ** (-m)
    assert hankel_transform(1 / r ** m, r, k, nu) == 2 * 2 ** (-m) * k ** (m - 2) * gamma(-m / 2 + nu / 2 + 1) / gamma(m / 2 + nu / 2)
    assert inverse_hankel_transform(2 ** (-m + 1) * k ** (m - 2) * gamma(-m / 2 + nu / 2 + 1) / gamma(m / 2 + nu / 2), k, r, nu) == r ** (-m)
    assert hankel_transform(r ** nu * exp(-a * r), r, k, nu) == 2 ** (nu + 1) * a * k ** (-nu - 3) * (a ** 2 / k ** 2 + 1) ** (-nu - S(3) / 2) * gamma(nu + Rational(3, 2)) / sqrt(pi)
    assert inverse_hankel_transform(2 ** (nu + 1) * a * k ** (-nu - 3) * (a ** 2 / k ** 2 + 1) ** (-nu - Rational(3, 2)) * gamma(nu + Rational(3, 2)) / sqrt(pi), k, r, nu) == r ** nu * exp(-a * r)