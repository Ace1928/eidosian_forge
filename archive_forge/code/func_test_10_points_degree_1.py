from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow
def test_10_points_degree_1():
    d = 1
    X = [-5, 2, 3, 4, 7, 9, 10, 30, 31, 34]
    Y = [-10, -2, 2, 4, 7, 6, 20, 45, 19, 25]
    spline = interpolating_spline(d, x, X, Y)
    assert spline == Piecewise((x * Rational(8, 7) - Rational(30, 7), (x >= -5) & (x <= 2)), (4 * x - 10, (x >= 2) & (x <= 3)), (2 * x - 4, (x >= 3) & (x <= 4)), (x, (x >= 4) & (x <= 7)), (-x / 2 + Rational(21, 2), (x >= 7) & (x <= 9)), (14 * x - 120, (x >= 9) & (x <= 10)), (x * Rational(5, 4) + Rational(15, 2), (x >= 10) & (x <= 30)), (-26 * x + 825, (x >= 30) & (x <= 31)), (2 * x - 43, (x >= 31) & (x <= 34)))