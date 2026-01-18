from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow
def test_basic_degree_0():
    d = 0
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    for i in range(len(splines)):
        assert splines[i] == Piecewise((1, Interval(i, i + 1).contains(x)), (0, True))