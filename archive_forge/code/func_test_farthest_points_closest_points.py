from sympy.core.function import (Derivative, Function)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions import exp, cos, sin, tan, cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Point, Point2D, Line, Polygon, Segment, convex_hull,\
from sympy.geometry.util import idiff, closest_points, farthest_points, _ordered_points, are_coplanar
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises
def test_farthest_points_closest_points():
    from sympy.core.random import randint
    from sympy.utilities.iterables import subsets
    for how in (min, max):
        if how == min:
            func = closest_points
        else:
            func = farthest_points
        raises(ValueError, lambda: func(Point2D(0, 0), Point2D(0, 0)))
        p1 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 1)]
        p2 = [Point2D(0, 0), Point2D(3, 0), Point2D(2, 1)]
        p3 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 10)]
        p4 = [Point2D(0, 0), Point2D(3, 0), Point2D(4, 0)]
        p5 = [Point2D(0, 0), Point2D(3, 0), Point2D(-1, 0)]
        dup = [Point2D(0, 0), Point2D(3, 0), Point2D(3, 0), Point2D(-1, 0)]
        x = Symbol('x', positive=True)
        s = [Point2D(a) for a in ((x, 1), (x + 3, 2), (x + 2, 2))]
        for points in (p1, p2, p3, p4, p5, dup, s):
            d = how((i.distance(j) for i, j in subsets(set(points), 2)))
            ans = a, b = list(func(*points))[0]
            assert a.distance(b) == d
            assert ans == _ordered_points(ans)
        points = set()
        while len(points) != 7:
            points.add(Point2D(randint(1, 100), randint(1, 100)))
        points = list(points)
        d = how((i.distance(j) for i, j in subsets(points, 2)))
        ans = a, b = list(func(*points))[0]
        assert a.distance(b) == d
        assert ans == _ordered_points(ans)
    a, b, c = (Point2D(0, 0), Point2D(1, 0), Point2D(S.Half, sqrt(3) / 2))
    ans = {_ordered_points((i, j)) for i, j in subsets((a, b, c), 2)}
    assert closest_points(b, c, a) == ans
    assert farthest_points(b, c, a) == ans
    points = [(1, 1), (1, 2), (3, 1), (-5, 2), (15, 4)]
    assert farthest_points(*points) == {(Point2D(-5, 2), Point2D(15, 4))}
    points = [(1, -1), (1, -2), (3, -1), (-5, -2), (15, -4)]
    assert farthest_points(*points) == {(Point2D(-5, -2), Point2D(15, -4))}
    assert farthest_points((1, 1), (0, 0)) == {(Point2D(0, 0), Point2D(1, 1))}
    raises(ValueError, lambda: farthest_points((1, 1)))