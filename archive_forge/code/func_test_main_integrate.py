from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_main_integrate():
    triangle = Polygon((0, 3), (5, 3), (1, 1))
    facets = triangle.sides
    hp_params = hyperplane_parameters(triangle)
    assert main_integrate(x ** 2 + y ** 2, facets, hp_params) == Rational(325, 6)
    assert main_integrate(x ** 2 + y ** 2, facets, hp_params, max_degree=1) == {0: 0, 1: 5, y: Rational(35, 3), x: 10}