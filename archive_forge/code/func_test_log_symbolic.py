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
def test_log_symbolic():
    assert log(x, exp(1)) == log(x)
    assert log(exp(x)) != x
    assert log(x, exp(1)) == log(x)
    assert log(x * y) != log(x) + log(y)
    assert log(x / y).expand() != log(x) - log(y)
    assert log(x / y).expand(force=True) == log(x) - log(y)
    assert log(x ** y).expand() != y * log(x)
    assert log(x ** y).expand(force=True) == y * log(x)
    assert log(x, 2) == log(x) / log(2)
    assert log(E, 2) == 1 / log(2)
    p, q = symbols('p,q', positive=True)
    r = Symbol('r', real=True)
    assert log(p ** 2) != 2 * log(p)
    assert log(p ** 2).expand() == 2 * log(p)
    assert log(x ** 2).expand() != 2 * log(x)
    assert log(p ** q) != q * log(p)
    assert log(exp(p)) == p
    assert log(p * q) != log(p) + log(q)
    assert log(p * q).expand() == log(p) + log(q)
    assert log(-sqrt(3)) == log(sqrt(3)) + I * pi
    assert log(-exp(p)) != p + I * pi
    assert log(-exp(x)).expand() != x + I * pi
    assert log(-exp(r)).expand() == r + I * pi
    assert log(x ** y) != y * log(x)
    assert (log(x ** (-5)) ** (-1)).expand() != -1 / log(x) / 5
    assert (log(p ** (-5)) ** (-1)).expand() == -1 / log(p) / 5
    assert log(-x).func is log and log(-x).args[0] == -x
    assert log(-p).func is log and log(-p).args[0] == -p