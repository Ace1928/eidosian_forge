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
def test_csc_rewrite():
    assert csc(x).rewrite(pow) == csc(x)
    assert csc(x).rewrite(sqrt) == csc(x)
    assert csc(x).rewrite(exp) == 2 * I / (exp(I * x) - exp(-I * x))
    assert csc(x).rewrite(sin) == 1 / sin(x)
    assert csc(x).rewrite(tan) == (tan(x / 2) ** 2 + 1) / (2 * tan(x / 2))
    assert csc(x).rewrite(cot) == (cot(x / 2) ** 2 + 1) / (2 * cot(x / 2))
    assert csc(x).rewrite(cos) == 1 / cos(x - pi / 2, evaluate=False)
    assert csc(x).rewrite(sec) == sec(-x + pi / 2, evaluate=False)
    assert csc(1 - exp(-besselj(I, I))).rewrite(cos) == -1 / cos(-pi / 2 - 1 + cos(I * besselj(I, I)) + I * cos(-pi / 2 + I * besselj(I, I), evaluate=False), evaluate=False)