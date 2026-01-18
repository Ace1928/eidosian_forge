from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow
def test_basic_degree_3():
    d = 3
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    b0 = Piecewise((x ** 3 / 6, Interval(0, 1).contains(x)), (Rational(2, 3) - 2 * x + 2 * x ** 2 - x ** 3 / 2, Interval(1, 2).contains(x)), (Rational(-22, 3) + 10 * x - 4 * x ** 2 + x ** 3 / 2, Interval(2, 3).contains(x)), (Rational(32, 3) - 8 * x + 2 * x ** 2 - x ** 3 / 6, Interval(3, 4).contains(x)), (0, True))
    assert splines[0] == b0