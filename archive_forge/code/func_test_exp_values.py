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
def test_exp_values():
    if global_parameters.exp_is_pow:
        assert type(exp(x)) is Pow
    else:
        assert type(exp(x)) is exp
    k = Symbol('k', integer=True)
    assert exp(nan) is nan
    assert exp(oo) is oo
    assert exp(-oo) == 0
    assert exp(0) == 1
    assert exp(1) == E
    assert exp(-1 + x).as_base_exp() == (S.Exp1, x - 1)
    assert exp(1 + x).as_base_exp() == (S.Exp1, x + 1)
    assert exp(pi * I / 2) == I
    assert exp(pi * I) == -1
    assert exp(pi * I * Rational(3, 2)) == -I
    assert exp(2 * pi * I) == 1
    assert refine(exp(pi * I * 2 * k)) == 1
    assert refine(exp(pi * I * 2 * (k + S.Half))) == -1
    assert refine(exp(pi * I * 2 * (k + Rational(1, 4)))) == I
    assert refine(exp(pi * I * 2 * (k + Rational(3, 4)))) == -I
    assert exp(log(x)) == x
    assert exp(2 * log(x)) == x ** 2
    assert exp(pi * log(x)) == x ** pi
    assert exp(17 * log(x) + E * log(y)) == x ** 17 * y ** E
    assert exp(x * log(x)) != x ** x
    assert exp(sin(x) * log(x)) != x
    assert exp(3 * log(x) + oo * x) == exp(oo * x) * x ** 3
    assert exp(4 * log(x) * log(y) + 3 * log(x)) == x ** 3 * exp(4 * log(x) * log(y))
    assert exp(-oo, evaluate=False).is_finite is True
    assert exp(oo, evaluate=False).is_finite is False