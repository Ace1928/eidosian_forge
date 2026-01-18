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
def test_asin_nseries():
    assert asin(x + 2)._eval_nseries(x, 4, None, I) == -asin(2) + pi + sqrt(3) * I * x / 3 - sqrt(3) * I * x ** 2 / 9 + sqrt(3) * I * x ** 3 / 18 + O(x ** 4)
    assert asin(x + 2)._eval_nseries(x, 4, None, -I) == asin(2) - sqrt(3) * I * x / 3 + sqrt(3) * I * x ** 2 / 9 - sqrt(3) * I * x ** 3 / 18 + O(x ** 4)
    assert asin(x - 2)._eval_nseries(x, 4, None, I) == -asin(2) - sqrt(3) * I * x / 3 - sqrt(3) * I * x ** 2 / 9 - sqrt(3) * I * x ** 3 / 18 + O(x ** 4)
    assert asin(x - 2)._eval_nseries(x, 4, None, -I) == asin(2) - pi + sqrt(3) * I * x / 3 + sqrt(3) * I * x ** 2 / 9 + sqrt(3) * I * x ** 3 / 18 + O(x ** 4)
    assert asin(1 + x)._eval_nseries(x, 3, None) == pi / 2 - sqrt(2) * sqrt(-x) - sqrt(2) * (-x) ** (S(3) / 2) / 12 - 3 * sqrt(2) * (-x) ** (S(5) / 2) / 160 + O(x ** 3)
    assert asin(-1 + x)._eval_nseries(x, 3, None) == -pi / 2 + sqrt(2) * sqrt(x) + sqrt(2) * x ** (S(3) / 2) / 12 + 3 * sqrt(2) * x ** (S(5) / 2) / 160 + O(x ** 3)
    assert asin(exp(x))._eval_nseries(x, 3, None) == pi / 2 - sqrt(2) * sqrt(-x) + sqrt(2) * (-x) ** (S(3) / 2) / 6 - sqrt(2) * (-x) ** (S(5) / 2) / 120 + O(x ** 3)
    assert asin(-exp(x))._eval_nseries(x, 3, None) == -pi / 2 + sqrt(2) * sqrt(-x) - sqrt(2) * (-x) ** (S(3) / 2) / 6 + sqrt(2) * (-x) ** (S(5) / 2) / 120 + O(x ** 3)