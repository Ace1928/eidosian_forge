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
@slow
def test_mellin_transform_bessel():
    from sympy.functions.elementary.miscellaneous import Max
    MT = mellin_transform
    assert MT(besselj(a, 2 * sqrt(x)), x, s) == (gamma(a / 2 + s) / gamma(a / 2 - s + 1), (-re(a) / 2, Rational(3, 4)), True)
    assert MT(sin(sqrt(x)) * besselj(a, sqrt(x)), x, s) == (2 ** a * gamma(-2 * s + S.Half) * gamma(a / 2 + s + S.Half) / (gamma(-a / 2 - s + 1) * gamma(a - 2 * s + 1)), (-re(a) / 2 - S.Half, Rational(1, 4)), True)
    assert MT(cos(sqrt(x)) * besselj(a, sqrt(x)), x, s) == (2 ** a * gamma(a / 2 + s) * gamma(-2 * s + S.Half) / (gamma(-a / 2 - s + S.Half) * gamma(a - 2 * s + 1)), (-re(a) / 2, Rational(1, 4)), True)
    assert MT(besselj(a, sqrt(x)) ** 2, x, s) == (gamma(a + s) * gamma(S.Half - s) / (sqrt(pi) * gamma(1 - s) * gamma(1 + a - s)), (-re(a), S.Half), True)
    assert MT(besselj(a, sqrt(x)) * besselj(-a, sqrt(x)), x, s) == (gamma(s) * gamma(S.Half - s) / (sqrt(pi) * gamma(1 - a - s) * gamma(1 + a - s)), (0, S.Half), True)
    assert MT(besselj(a - 1, sqrt(x)) * besselj(a, sqrt(x)), x, s) == (gamma(1 - s) * gamma(a + s - S.Half) / (sqrt(pi) * gamma(Rational(3, 2) - s) * gamma(a - s + S.Half)), (S.Half - re(a), S.Half), True)
    assert MT(besselj(a, sqrt(x)) * besselj(b, sqrt(x)), x, s) == (4 ** s * gamma(1 - 2 * s) * gamma((a + b) / 2 + s) / (gamma(1 - s + (b - a) / 2) * gamma(1 - s + (a - b) / 2) * gamma(1 - s + (a + b) / 2)), (-(re(a) + re(b)) / 2, S.Half), True)
    assert MT(besselj(a, sqrt(x)) ** 2 + besselj(-a, sqrt(x)) ** 2, x, s)[1:] == ((Max(re(a), -re(a)), S.Half), True)
    assert MT(bessely(a, 2 * sqrt(x)), x, s) == (-cos(pi * (a / 2 - s)) * gamma(s - a / 2) * gamma(s + a / 2) / pi, (Max(-re(a) / 2, re(a) / 2), Rational(3, 4)), True)
    assert MT(sin(sqrt(x)) * bessely(a, sqrt(x)), x, s) == (-4 ** s * sin(pi * (a / 2 - s)) * gamma(S.Half - 2 * s) * gamma((1 - a) / 2 + s) * gamma((1 + a) / 2 + s) / (sqrt(pi) * gamma(1 - s - a / 2) * gamma(1 - s + a / 2)), (Max(-(re(a) + 1) / 2, (re(a) - 1) / 2), Rational(1, 4)), True)
    assert MT(cos(sqrt(x)) * bessely(a, sqrt(x)), x, s) == (-4 ** s * cos(pi * (a / 2 - s)) * gamma(s - a / 2) * gamma(s + a / 2) * gamma(S.Half - 2 * s) / (sqrt(pi) * gamma(S.Half - s - a / 2) * gamma(S.Half - s + a / 2)), (Max(-re(a) / 2, re(a) / 2), Rational(1, 4)), True)
    assert MT(besselj(a, sqrt(x)) * bessely(a, sqrt(x)), x, s) == (-cos(pi * s) * gamma(s) * gamma(a + s) * gamma(S.Half - s) / (pi ** S('3/2') * gamma(1 + a - s)), (Max(-re(a), 0), S.Half), True)
    assert MT(besselj(a, sqrt(x)) * bessely(b, sqrt(x)), x, s) == (-4 ** s * cos(pi * (a / 2 - b / 2 + s)) * gamma(1 - 2 * s) * gamma(a / 2 - b / 2 + s) * gamma(a / 2 + b / 2 + s) / (pi * gamma(a / 2 - b / 2 - s + 1) * gamma(a / 2 + b / 2 - s + 1)), (Max((-re(a) + re(b)) / 2, (-re(a) - re(b)) / 2), S.Half), True)
    assert MT(bessely(a, sqrt(x)) ** 2, x, s)[1:] == ((Max(-re(a), 0, re(a)), S.Half), True)
    assert MT(besselk(a, 2 * sqrt(x)), x, s) == (gamma(s - a / 2) * gamma(s + a / 2) / 2, (Max(-re(a) / 2, re(a) / 2), oo), True)
    assert MT(besselj(a, 2 * sqrt(2 * sqrt(x))) * besselk(a, 2 * sqrt(2 * sqrt(x))), x, s) == (4 ** (-s) * gamma(2 * s) * gamma(a / 2 + s) / (2 * gamma(a / 2 - s + 1)), (Max(0, -re(a) / 2), oo), True)
    assert MT(besseli(a, sqrt(x)) * besselk(a, sqrt(x)), x, s) == (gamma(s) * gamma(a + s) * gamma(-s + S.Half) / (2 * sqrt(pi) * gamma(a - s + 1)), (Max(-re(a), 0), S.Half), True)
    assert MT(besseli(b, sqrt(x)) * besselk(a, sqrt(x)), x, s) == (2 ** (2 * s - 1) * gamma(-2 * s + 1) * gamma(-a / 2 + b / 2 + s) * gamma(a / 2 + b / 2 + s) / (gamma(-a / 2 + b / 2 - s + 1) * gamma(a / 2 + b / 2 - s + 1)), (Max(-re(a) / 2 - re(b) / 2, re(a) / 2 - re(b) / 2), S.Half), True)
    mt = MT(exp(-x / 2) * besselk(a, x / 2), x, s)
    mt0 = gammasimp(trigsimp(gammasimp(mt[0].expand(func=True))))
    assert mt0 == 2 * pi ** Rational(3, 2) * cos(pi * s) * gamma(S.Half - s) / ((cos(2 * pi * a) - cos(2 * pi * s)) * gamma(-a - s + 1) * gamma(a - s + 1))
    assert mt[1:] == ((Max(-re(a), re(a)), oo), True)