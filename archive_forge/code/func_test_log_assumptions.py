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
def test_log_assumptions():
    p = symbols('p', positive=True)
    n = symbols('n', negative=True)
    z = symbols('z', zero=True)
    x = symbols('x', infinite=True, extended_positive=True)
    assert log(z).is_positive is False
    assert log(x).is_extended_positive is True
    assert log(2) > 0
    assert log(1, evaluate=False).is_zero
    assert log(1 + z).is_zero
    assert log(p).is_zero is None
    assert log(n).is_zero is False
    assert log(0.5).is_negative is True
    assert log(exp(p) + 1).is_positive
    assert log(1, evaluate=False).is_algebraic
    assert log(42, evaluate=False).is_algebraic is False
    assert log(1 + z).is_rational