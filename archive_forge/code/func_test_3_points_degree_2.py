from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow
def test_3_points_degree_2():
    d = 2
    X = [-3, 10, 19]
    Y = [3, -4, 30]
    spline = interpolating_spline(d, x, X, Y)
    assert spline == Piecewise((505 * x ** 2 / 2574 - x * Rational(4921, 2574) - Rational(1931, 429), (x >= -3) & (x <= 19)))