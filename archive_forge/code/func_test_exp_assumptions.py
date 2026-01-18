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
def test_exp_assumptions():
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    for e in (exp, exp_polar):
        assert e(x).is_real is None
        assert e(x).is_imaginary is None
        assert e(i).is_real is None
        assert e(i).is_imaginary is None
        assert e(r).is_real is True
        assert e(r).is_imaginary is False
        assert e(re(x)).is_extended_real is True
        assert e(re(x)).is_imaginary is False
    assert Pow(E, I * pi, evaluate=False).is_imaginary == False
    assert Pow(E, 2 * I * pi, evaluate=False).is_imaginary == False
    assert Pow(E, I * pi / 2, evaluate=False).is_imaginary == True
    assert Pow(E, I * pi / 3, evaluate=False).is_imaginary is None
    assert exp(0, evaluate=False).is_algebraic
    a = Symbol('a', algebraic=True)
    an = Symbol('an', algebraic=True, nonzero=True)
    r = Symbol('r', rational=True)
    rn = Symbol('rn', rational=True, nonzero=True)
    assert exp(a).is_algebraic is None
    assert exp(an).is_algebraic is False
    assert exp(pi * r).is_algebraic is None
    assert exp(pi * rn).is_algebraic is False
    assert exp(0, evaluate=False).is_algebraic is True
    assert exp(I * pi / 3, evaluate=False).is_algebraic is True
    assert exp(I * pi * r, evaluate=False).is_algebraic is True