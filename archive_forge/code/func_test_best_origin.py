from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_best_origin():
    expr1 = y ** 2 * x ** 5 + y ** 5 * x ** 7 + 7 * x + x ** 12 + y ** 7 * x
    l1 = Segment2D(Point(0, 3), Point(1, 1))
    l2 = Segment2D(Point(S(3) / 2, 0), Point(S(3) / 2, 3))
    l3 = Segment2D(Point(0, S(3) / 2), Point(3, S(3) / 2))
    l4 = Segment2D(Point(0, 2), Point(2, 0))
    l5 = Segment2D(Point(0, 2), Point(1, 1))
    l6 = Segment2D(Point(2, 0), Point(1, 1))
    assert best_origin((2, 1), 3, l1, expr1) == (0, 3)
    assert best_origin((2, 0), 3, l2, x ** 7) == (1.5, 0)
    assert best_origin((0, 2), 3, l3, x ** 7) == (0, 1.5)
    assert best_origin((1, 1), 2, l4, x ** 7 * y ** 3) == (0, 2)
    assert best_origin((1, 1), 2, l4, x ** 3 * y ** 7) == (2, 0)
    assert best_origin((1, 1), 2, l5, x ** 2 * y ** 9) == (0, 2)
    assert best_origin((1, 1), 2, l6, x ** 9 * y ** 2) == (2, 0)