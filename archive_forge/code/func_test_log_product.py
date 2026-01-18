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
def test_log_product():
    from sympy.abc import n, m
    i, j = symbols('i,j', positive=True, integer=True)
    x, y = symbols('x,y', positive=True)
    z = symbols('z', real=True)
    w = symbols('w')
    expr = log(Product(x ** i, (i, 1, n)))
    assert simplify(expr) == expr
    assert expr.expand() == Sum(i * log(x), (i, 1, n))
    expr = log(Product(x ** i * y ** j, (i, 1, n), (j, 1, m)))
    assert simplify(expr) == expr
    assert expr.expand() == Sum(i * log(x) + j * log(y), (i, 1, n), (j, 1, m))
    expr = log(Product(-2, (n, 0, 4)))
    assert simplify(expr) == expr
    assert expr.expand() == expr
    assert expr.expand(force=True) == Sum(log(-2), (n, 0, 4))
    expr = log(Product(exp(z * i), (i, 0, n)))
    assert expr.expand() == Sum(z * i, (i, 0, n))
    expr = log(Product(exp(w * i), (i, 0, n)))
    assert expr.expand() == expr
    assert expr.expand(force=True) == Sum(w * i, (i, 0, n))
    expr = log(Product(i ** 2 * abs(j), (i, 1, n), (j, 1, m)))
    assert expr.expand() == Sum(2 * log(i) + log(j), (i, 1, n), (j, 1, m))