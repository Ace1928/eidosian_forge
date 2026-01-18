from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_integration_reduction():
    triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    facets = triangle.sides
    a, b = hyperplane_parameters(triangle)[0]
    assert integration_reduction(facets, 0, a, b, 1, (x, y), 0) == 5
    assert integration_reduction(facets, 0, a, b, 0, (x, y), 0) == 0