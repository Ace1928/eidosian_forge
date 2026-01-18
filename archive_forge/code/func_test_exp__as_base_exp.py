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
def test_exp__as_base_exp():
    assert exp(x).as_base_exp() == (E, x)
    assert exp(2 * x).as_base_exp() == (E, 2 * x)
    assert exp(x * y).as_base_exp() == (E, x * y)
    assert exp(-x).as_base_exp() == (E, -x)
    assert E ** x == exp(x)
    assert E ** (2 * x) == exp(2 * x)
    assert E ** (x * y) == exp(x * y)
    assert exp(x).base is S.Exp1
    assert exp(x).exp == x