from sympy.core.function import (Derivative, Function)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions import exp, cos, sin, tan, cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Point, Point2D, Line, Polygon, Segment, convex_hull,\
from sympy.geometry.util import idiff, closest_points, farthest_points, _ordered_points, are_coplanar
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises
def test_are_coplanar():
    a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    d = Line(Point2D(0, 3), Point2D(1, 5))
    assert are_coplanar(a, b, c) == False
    assert are_coplanar(a, d) == False