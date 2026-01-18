from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises
def test_issue_4420():
    i = Symbol('i', integer=True)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)
    assert cos(4 * i * pi) == 1
    assert sin(4 * i * pi) == 0
    assert tan(4 * i * pi) == 0
    assert cot(4 * i * pi) is zoo
    assert cos(3 * i * pi) == cos(pi * i)
    assert sin(3 * i * pi) == 0
    assert tan(3 * i * pi) == 0
    assert cot(3 * i * pi) is zoo
    assert cos(4.0 * i * pi) == 1
    assert sin(4.0 * i * pi) == 0
    assert tan(4.0 * i * pi) == 0
    assert cot(4.0 * i * pi) is zoo
    assert cos(3.0 * i * pi) == cos(pi * i)
    assert sin(3.0 * i * pi) == 0
    assert tan(3.0 * i * pi) == 0
    assert cot(3.0 * i * pi) is zoo
    assert cos(4.5 * i * pi) == cos(0.5 * pi * i)
    assert sin(4.5 * i * pi) == sin(0.5 * pi * i)
    assert tan(4.5 * i * pi) == tan(0.5 * pi * i)
    assert cot(4.5 * i * pi) == cot(0.5 * pi * i)
    assert cos(4 * e * pi) == 1
    assert sin(4 * e * pi) == 0
    assert tan(4 * e * pi) == 0
    assert cot(4 * e * pi) is zoo
    assert cos(3 * e * pi) == 1
    assert sin(3 * e * pi) == 0
    assert tan(3 * e * pi) == 0
    assert cot(3 * e * pi) is zoo
    assert cos(4.0 * e * pi) == 1
    assert sin(4.0 * e * pi) == 0
    assert tan(4.0 * e * pi) == 0
    assert cot(4.0 * e * pi) is zoo
    assert cos(3.0 * e * pi) == 1
    assert sin(3.0 * e * pi) == 0
    assert tan(3.0 * e * pi) == 0
    assert cot(3.0 * e * pi) is zoo
    assert cos(4.5 * e * pi) == cos(0.5 * pi * e)
    assert sin(4.5 * e * pi) == sin(0.5 * pi * e)
    assert tan(4.5 * e * pi) == tan(0.5 * pi * e)
    assert cot(4.5 * e * pi) == cot(0.5 * pi * e)
    assert cos(4 * o * pi) == 1
    assert sin(4 * o * pi) == 0
    assert tan(4 * o * pi) == 0
    assert cot(4 * o * pi) is zoo
    assert cos(3 * o * pi) == -1
    assert sin(3 * o * pi) == 0
    assert tan(3 * o * pi) == 0
    assert cot(3 * o * pi) is zoo
    assert cos(4.0 * o * pi) == 1
    assert sin(4.0 * o * pi) == 0
    assert tan(4.0 * o * pi) == 0
    assert cot(4.0 * o * pi) is zoo
    assert cos(3.0 * o * pi) == -1
    assert sin(3.0 * o * pi) == 0
    assert tan(3.0 * o * pi) == 0
    assert cot(3.0 * o * pi) is zoo
    assert cos(4.5 * o * pi) == cos(0.5 * pi * o)
    assert sin(4.5 * o * pi) == sin(0.5 * pi * o)
    assert tan(4.5 * o * pi) == tan(0.5 * pi * o)
    assert cot(4.5 * o * pi) == cot(0.5 * pi * o)
    assert cos(4 * x * pi) == cos(4 * pi * x)
    assert sin(4 * x * pi) == sin(4 * pi * x)
    assert tan(4 * x * pi) == tan(4 * pi * x)
    assert cot(4 * x * pi) == cot(4 * pi * x)
    assert cos(3 * x * pi) == cos(3 * pi * x)
    assert sin(3 * x * pi) == sin(3 * pi * x)
    assert tan(3 * x * pi) == tan(3 * pi * x)
    assert cot(3 * x * pi) == cot(3 * pi * x)
    assert cos(4.0 * x * pi) == cos(4.0 * pi * x)
    assert sin(4.0 * x * pi) == sin(4.0 * pi * x)
    assert tan(4.0 * x * pi) == tan(4.0 * pi * x)
    assert cot(4.0 * x * pi) == cot(4.0 * pi * x)
    assert cos(3.0 * x * pi) == cos(3.0 * pi * x)
    assert sin(3.0 * x * pi) == sin(3.0 * pi * x)
    assert tan(3.0 * x * pi) == tan(3.0 * pi * x)
    assert cot(3.0 * x * pi) == cot(3.0 * pi * x)
    assert cos(4.5 * x * pi) == cos(4.5 * pi * x)
    assert sin(4.5 * x * pi) == sin(4.5 * pi * x)
    assert tan(4.5 * x * pi) == tan(4.5 * pi * x)
    assert cot(4.5 * x * pi) == cot(4.5 * pi * x)