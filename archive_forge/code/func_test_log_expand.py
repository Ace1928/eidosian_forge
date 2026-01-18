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
def test_log_expand():
    w = Symbol('w', positive=True)
    e = log(w ** (log(5) / log(3)))
    assert e.expand() == log(5) / log(3) * log(w)
    x, y, z = symbols('x,y,z', positive=True)
    assert log(x * (y + z)).expand(mul=False) == log(x) + log(y + z)
    assert log(log(x ** 2) * log(y * z)).expand() in [log(2 * log(x) * log(y) + 2 * log(x) * log(z)), log(log(x) * log(z) + log(y) * log(x)) + log(2), log((log(y) + log(z)) * log(x)) + log(2)]
    assert log(x ** log(x ** 2)).expand(deep=False) == log(x) * log(x ** 2)
    assert log(x ** log(x ** 2)).expand() == 2 * log(x) ** 2
    x, y = symbols('x,y')
    assert log(x * y).expand(force=True) == log(x) + log(y)
    assert log(x ** y).expand(force=True) == y * log(x)
    assert log(exp(x)).expand(force=True) == x
    assert log(2 * 3 ** 2).expand() != 2 * log(3) + log(2)