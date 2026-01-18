from sympy.concrete.summations import Sum
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import clear_cache
from sympy.core.containers import Dict, Tuple
from sympy.core.expr import Expr, unchanged
from sympy.core.function import (Subs, Function, diff, Lambda, expand,
from sympy.core.numbers import E, Float, zoo, Rational, pi, I, oo, nan
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Dummy, Symbol
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, acos
from sympy.functions.special.error_functions import expint
from sympy.functions.special.gamma_functions import loggamma, polygamma
from sympy.matrices.dense import Matrix
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.tensor.indexed import Indexed
from sympy.core.function import (PoleError, _mexpand, arity,
from sympy.core.parameters import _exp_is_pow
from sympy.core.sympify import sympify, SympifyError
from sympy.matrices import MutableMatrix, ImmutableMatrix
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import subsets, variations
from sympy.testing.pytest import XFAIL, raises, warns_deprecated_sympy, _both_exp_pow
from sympy.abc import t, w, x, y, z
def test_Derivative_as_finite_difference():
    x, h = symbols('x h', real=True)
    dfdx = f(x).diff(x)
    assert (dfdx.as_finite_difference([x - 2, x - 1, x, x + 1, x + 2]) - (S.One / 12 * (f(x - 2) - f(x + 2)) + Rational(2, 3) * (f(x + 1) - f(x - 1)))).simplify() == 0
    assert (dfdx.as_finite_difference() - (f(x + S.Half) - f(x - S.Half))).simplify() == 0
    assert (dfdx.as_finite_difference(h) - (f(x + h / S(2)) - f(x - h / S(2))) / h).simplify() == 0
    assert (dfdx.as_finite_difference([x - 3 * h, x - h, x + h, x + 3 * h]) - (S(9) / (8 * 2 * h) * (f(x + h) - f(x - h)) + S.One / (24 * 2 * h) * (f(x - 3 * h) - f(x + 3 * h)))).simplify() == 0
    assert (dfdx.as_finite_difference([0, 1, 2], 0) - (Rational(-3, 2) * f(0) + 2 * f(1) - f(2) / 2)).simplify() == 0
    assert (dfdx.as_finite_difference([x, x + h], x) - (f(x + h) - f(x)) / h).simplify() == 0
    assert (dfdx.as_finite_difference([x - h, x, x + h], x - h) - (-S(3) / (2 * h) * f(x - h) + 2 / h * f(x) - S.One / (2 * h) * f(x + h))).simplify() == 0
    assert (dfdx.as_finite_difference([x - h, x + h, x + 3 * h, x + 5 * h, x + 7 * h]) - 1 / (2 * h) * (-S(11) / 12 * f(x - h) + S(17) / 24 * f(x + h) + Rational(3, 8) * f(x + 3 * h) - Rational(5, 24) * f(x + 5 * h) + S.One / 24 * f(x + 7 * h))).simplify() == 0
    d2fdx2 = f(x).diff(x, 2)
    assert (d2fdx2.as_finite_difference([x - h, x, x + h]) - h ** (-2) * (f(x - h) + f(x + h) - 2 * f(x))).simplify() == 0
    assert (d2fdx2.as_finite_difference([x - 2 * h, x - h, x, x + h, x + 2 * h]) - h ** (-2) * (Rational(-1, 12) * (f(x - 2 * h) + f(x + 2 * h)) + Rational(4, 3) * (f(x + h) + f(x - h)) - Rational(5, 2) * f(x))).simplify() == 0
    assert (d2fdx2.as_finite_difference([x - 3 * h, x - h, x + h, x + 3 * h]) - (2 * h) ** (-2) * (S.Half * (f(x - 3 * h) + f(x + 3 * h)) - S.Half * (f(x + h) + f(x - h)))).simplify() == 0
    assert (d2fdx2.as_finite_difference([x, x + h, x + 2 * h, x + 3 * h]) - h ** (-2) * (2 * f(x) - 5 * f(x + h) + 4 * f(x + 2 * h) - f(x + 3 * h))).simplify() == 0
    assert (d2fdx2.as_finite_difference([x - h, x + h, x + 3 * h, x + 5 * h]) - (2 * h) ** (-2) * (Rational(3, 2) * f(x - h) - Rational(7, 2) * f(x + h) + Rational(5, 2) * f(x + 3 * h) - S.Half * f(x + 5 * h))).simplify() == 0
    d3fdx3 = f(x).diff(x, 3)
    assert (d3fdx3.as_finite_difference() - (-f(x - Rational(3, 2)) + 3 * f(x - S.Half) - 3 * f(x + S.Half) + f(x + Rational(3, 2)))).simplify() == 0
    assert (d3fdx3.as_finite_difference([x - 3 * h, x - 2 * h, x - h, x, x + h, x + 2 * h, x + 3 * h]) - h ** (-3) * (S.One / 8 * (f(x - 3 * h) - f(x + 3 * h)) - f(x - 2 * h) + f(x + 2 * h) + Rational(13, 8) * (f(x - h) - f(x + h)))).simplify() == 0
    assert (d3fdx3.as_finite_difference([x - 3 * h, x - h, x + h, x + 3 * h]) - (2 * h) ** (-3) * (f(x + 3 * h) - f(x - 3 * h) + 3 * (f(x - h) - f(x + h)))).simplify() == 0
    assert (d3fdx3.as_finite_difference([x, x + h, x + 2 * h, x + 3 * h]) - h ** (-3) * (f(x + 3 * h) - f(x) + 3 * (f(x + h) - f(x + 2 * h)))).simplify() == 0
    assert (d3fdx3.as_finite_difference([x - h, x + h, x + 3 * h, x + 5 * h]) - (2 * h) ** (-3) * (f(x + 5 * h) - f(x - h) + 3 * (f(x + h) - f(x + 3 * h)))).simplify() == 0
    y = Symbol('y', real=True)
    d2fdxdy = f(x, y).diff(x, y)
    ref0 = Derivative(f(x + S.Half, y), y) - Derivative(f(x - S.Half, y), y)
    assert (d2fdxdy.as_finite_difference(wrt=x) - ref0).simplify() == 0
    half = S.Half
    xm, xp, ym, yp = (x - half, x + half, y - half, y + half)
    ref2 = f(xm, ym) + f(xp, yp) - f(xp, ym) - f(xm, yp)
    assert (d2fdxdy.as_finite_difference() - ref2).simplify() == 0