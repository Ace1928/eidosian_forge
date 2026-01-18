from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow
@slow
def test_6_points_degree_3():
    d = 3
    X = [-1, 0, 2, 3, 9, 12]
    Y = [-4, 3, 3, 7, 9, 20]
    spline = interpolating_spline(d, x, X, Y)
    assert spline == Piecewise((6058 * x ** 3 / 5301 - 18427 * x ** 2 / 5301 + x * Rational(12622, 5301) + 3, (x >= -1) & (x <= 2)), (-8327 * x ** 3 / 5301 + 67883 * x ** 2 / 5301 - x * Rational(159998, 5301) + Rational(43661, 1767), (x >= 2) & (x <= 3)), (5414 * x ** 3 / 47709 - 1386 * x ** 2 / 589 + x * Rational(4267, 279) - Rational(12232, 589), (x >= 3) & (x <= 12)))