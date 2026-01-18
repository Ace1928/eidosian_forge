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
def test_exp_rewrite():
    assert exp(x).rewrite(sin) == sinh(x) + cosh(x)
    assert exp(x * I).rewrite(cos) == cos(x) + I * sin(x)
    assert exp(1).rewrite(cos) == sinh(1) + cosh(1)
    assert exp(1).rewrite(sin) == sinh(1) + cosh(1)
    assert exp(1).rewrite(sin) == sinh(1) + cosh(1)
    assert exp(x).rewrite(tanh) == (1 + tanh(x / 2)) / (1 - tanh(x / 2))
    assert exp(pi * I / 4).rewrite(sqrt) == sqrt(2) / 2 + sqrt(2) * I / 2
    assert exp(pi * I / 3).rewrite(sqrt) == S.Half + sqrt(3) * I / 2
    if not global_parameters.exp_is_pow:
        assert exp(x * log(y)).rewrite(Pow) == y ** x
        assert exp(log(x) * log(y)).rewrite(Pow) in [x ** log(y), y ** log(x)]
        assert exp(log(log(x)) * y).rewrite(Pow) == log(x) ** y
    n = Symbol('n', integer=True)
    assert Sum((exp(pi * I / 2) / 2) ** n, (n, 0, oo)).rewrite(sqrt).doit() == Rational(4, 5) + I * 2 / 5
    assert Sum((exp(pi * I / 4) / 2) ** n, (n, 0, oo)).rewrite(sqrt).doit() == 1 / (1 - sqrt(2) * (1 + I) / 4)
    assert Sum((exp(pi * I / 3) / 2) ** n, (n, 0, oo)).rewrite(sqrt).doit().cancel() == 4 * I / (sqrt(3) + 3 * I)