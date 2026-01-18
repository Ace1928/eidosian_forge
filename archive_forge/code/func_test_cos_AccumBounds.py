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
def test_cos_AccumBounds():
    assert cos(AccumBounds(-oo, oo)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(0, oo)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(-oo, 0)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(0, 2 * S.Pi)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(-S.Pi / 3, S.Pi / 4)) == AccumBounds(cos(-S.Pi / 3), 1)
    assert cos(AccumBounds(S.Pi * Rational(3, 4), S.Pi * Rational(5, 4))) == AccumBounds(-1, cos(S.Pi * Rational(3, 4)))
    assert cos(AccumBounds(S.Pi * Rational(5, 4), S.Pi * Rational(4, 3))) == AccumBounds(cos(S.Pi * Rational(5, 4)), cos(S.Pi * Rational(4, 3)))
    assert cos(AccumBounds(S.Pi / 4, S.Pi / 3)) == AccumBounds(cos(S.Pi / 3), cos(S.Pi / 4))