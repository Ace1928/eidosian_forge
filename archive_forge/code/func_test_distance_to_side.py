from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_distance_to_side():
    point = (0, 0, 0)
    assert distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0)) == -sqrt(2) / 2