from sympy.assumptions.refine import refine
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import expand_log
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, re, sign, transpose)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.polytools import gcd
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.parameters import global_parameters
from sympy.functions.elementary.exponential import match_real_imag
from sympy.abc import x, y, z
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises, XFAIL, _both_exp_pow
@_both_exp_pow
def test_exp_period():
    assert exp(I * pi * Rational(9, 4)) == exp(I * pi / 4)
    assert exp(I * pi * Rational(46, 18)) == exp(I * pi * Rational(5, 9))
    assert exp(I * pi * Rational(25, 7)) == exp(I * pi * Rational(-3, 7))
    assert exp(I * pi * Rational(-19, 3)) == exp(-I * pi / 3)
    assert exp(I * pi * Rational(37, 8)) - exp(I * pi * Rational(-11, 8)) == 0
    assert exp(I * pi * Rational(-5, 3)) / exp(I * pi * Rational(11, 5)) * exp(I * pi * Rational(148, 15)) == 1
    assert exp(2 - I * pi * Rational(17, 5)) == exp(2 + I * pi * Rational(3, 5))
    assert exp(log(3) + I * pi * Rational(29, 9)) == 3 * exp(I * pi * Rational(-7, 9))
    n = Symbol('n', integer=True)
    e = Symbol('e', even=True)
    assert exp(e * I * pi) == 1
    assert exp((e + 1) * I * pi) == -1
    assert exp((1 + 4 * n) * I * pi / 2) == I
    assert exp((-1 + 4 * n) * I * pi / 2) == -I