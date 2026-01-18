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
def test_asec_leading_term():
    assert asec(1 / x).as_leading_term(x) == pi / 2
    assert asec(x + 1).as_leading_term(x) == sqrt(2) * sqrt(x)
    assert asec(x - 1).as_leading_term(x) == pi
    assert asec(x).as_leading_term(x, cdir=1) == -I * log(x) + I * log(2)
    assert asec(x).as_leading_term(x, cdir=-1) == I * log(x) + 2 * pi - I * log(2)
    assert asec(I * x + 1 / 2).as_leading_term(x, cdir=1) == asec(1 / 2)
    assert asec(-I * x + 1 / 2).as_leading_term(x, cdir=1) == -asec(1 / 2)
    assert asec(I * x - 1 / 2).as_leading_term(x, cdir=1) == 2 * pi - asec(-1 / 2)
    assert asec(-I * x - 1 / 2).as_leading_term(x, cdir=1) == asec(-1 / 2)
    assert asec(-I * x ** 2 + x - S(1) / 2).as_leading_term(x, cdir=1) == pi + I * log(2 - sqrt(3))
    assert asec(-I * x ** 2 + x - S(1) / 2).as_leading_term(x, cdir=-1) == pi + I * log(2 - sqrt(3))