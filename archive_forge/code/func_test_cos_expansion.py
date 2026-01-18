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
def test_cos_expansion():
    assert cos(x + y).expand(trig=True) == cos(x) * cos(y) - sin(x) * sin(y)
    assert cos(x - y).expand(trig=True) == cos(x) * cos(y) + sin(x) * sin(y)
    assert cos(y - x).expand(trig=True) == cos(x) * cos(y) + sin(x) * sin(y)
    assert cos(2 * x).expand(trig=True) == 2 * cos(x) ** 2 - 1
    assert cos(3 * x).expand(trig=True) == 4 * cos(x) ** 3 - 3 * cos(x)
    assert cos(4 * x).expand(trig=True) == 8 * cos(x) ** 4 - 8 * cos(x) ** 2 + 1
    _test_extrig(cos, 2, 2 * cos(1) ** 2 - 1)
    _test_extrig(cos, 3, 4 * cos(1) ** 3 - 3 * cos(1))