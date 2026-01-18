import math
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.evalf import N
from sympy.core.function import (Function, nfloat)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio)
from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Rational,
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.complexes import (Abs, re, im)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, cosh)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.printing import srepr
from sympy.printing.str import sstr
from sympy.simplify.simplify import simplify
from sympy.core.numbers import comp
from sympy.core.evalf import (complex_accuracy, PrecisionExhausted,
from mpmath import inf, ninf, make_mpc
from mpmath.libmp.libmpf import from_float, fzero
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import n, x, y
def test_evalf_trig():
    assert NS('sin(1)', 15) == '0.841470984807897'
    assert NS('cos(1)', 15) == '0.540302305868140'
    assert NS('sin(10**-6)', 15) == '9.99999999999833e-7'
    assert NS('cos(10**-6)', 15) == '0.999999999999500'
    assert NS('sin(E*10**100)', 15) == '0.409160531722613'
    assert NS(sin(exp(pi * sqrt(163)) * pi), 15) == '-2.35596641936785e-12'
    assert NS(sin(pi * 10 ** 100 + Rational(7, 10 ** 5), evaluate=False), 15, maxn=120) == '6.99999999428333e-5'
    assert NS(sin(Rational(7, 10 ** 5), evaluate=False), 15) == '6.99999999428333e-5'