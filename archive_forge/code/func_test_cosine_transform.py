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
def test_cosine_transform():
    from sympy.functions.special.error_functions import Ci, Si
    t = symbols('t')
    w = symbols('w')
    a = symbols('a')
    f = Function('f')
    assert cosine_transform(f(t), t, w) == CosineTransform(f(t), t, w)
    assert inverse_cosine_transform(f(w), w, t) == InverseCosineTransform(f(w), w, t)
    assert cosine_transform(1 / sqrt(t), t, w) == 1 / sqrt(w)
    assert inverse_cosine_transform(1 / sqrt(w), w, t) == 1 / sqrt(t)
    assert cosine_transform(1 / (a ** 2 + t ** 2), t, w) == sqrt(2) * sqrt(pi) * exp(-a * w) / (2 * a)
    assert cosine_transform(t ** (-a), t, w) == 2 ** (-a + S.Half) * w ** (a - 1) * gamma((-a + 1) / 2) / gamma(a / 2)
    assert inverse_cosine_transform(2 ** (-a + S(1) / 2) * w ** (a - 1) * gamma(-a / 2 + S.Half) / gamma(a / 2), w, t) == t ** (-a)
    assert cosine_transform(exp(-a * t), t, w) == sqrt(2) * a / (sqrt(pi) * (a ** 2 + w ** 2))
    assert inverse_cosine_transform(sqrt(2) * a / (sqrt(pi) * (a ** 2 + w ** 2)), w, t) == exp(-a * t)
    assert cosine_transform(exp(-a * sqrt(t)) * cos(a * sqrt(t)), t, w) == a * exp(-a ** 2 / (2 * w)) / (2 * w ** Rational(3, 2))
    assert cosine_transform(1 / (a + t), t, w) == sqrt(2) * ((-2 * Si(a * w) + pi) * sin(a * w) / 2 - cos(a * w) * Ci(a * w)) / sqrt(pi)
    assert inverse_cosine_transform(sqrt(2) * meijerg(((S.Half, 0), ()), ((S.Half, 0, 0), (S.Half,)), a ** 2 * w ** 2 / 4) / (2 * pi), w, t) == 1 / (a + t)
    assert cosine_transform(1 / sqrt(a ** 2 + t ** 2), t, w) == sqrt(2) * meijerg(((S.Half,), ()), ((0, 0), (S.Half,)), a ** 2 * w ** 2 / 4) / (2 * sqrt(pi))
    assert inverse_cosine_transform(sqrt(2) * meijerg(((S.Half,), ()), ((0, 0), (S.Half,)), a ** 2 * w ** 2 / 4) / (2 * sqrt(pi)), w, t) == 1 / (t * sqrt(a ** 2 / t ** 2 + 1))