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
def test_cot_series():
    assert cot(x).series(x, 0, 9) == 1 / x - x / 3 - x ** 3 / 45 - 2 * x ** 5 / 945 - x ** 7 / 4725 + O(x ** 9)
    assert cot(x ** 4 + x ** 5).series(x, 0, 1) == x ** (-4) - 1 / x ** 3 + x ** (-2) - 1 / x + 1 + O(x)
    assert cot(pi * (1 - x)).series(x, 0, 3) == -1 / (pi * x) + pi * x / 3 + O(x ** 3)
    assert cot(x).taylor_term(0, x) == 1 / x
    assert cot(x).taylor_term(2, x) is S.Zero
    assert cot(x).taylor_term(3, x) == -x ** 3 / 45