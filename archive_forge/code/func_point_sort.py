from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def point_sort(poly, normal=None, clockwise=True):
    """Returns the same polygon with points sorted in clockwise or
    anti-clockwise order.

    Note that it's necessary for input points to be sorted in some order
    (clockwise or anti-clockwise) for the integration algorithm to work.
    As a convention algorithm has been implemented keeping clockwise
    orientation in mind.

    Parameters
    ==========

    poly:
        2D or 3D Polygon.
    normal : optional
        The normal of the plane which the 3-Polytope is a part of.
    clockwise : bool, optional
        Returns points sorted in clockwise order if True and
        anti-clockwise if False.

    Examples
    ========

    >>> from sympy.integrals.intpoly import point_sort
    >>> from sympy import Point
    >>> point_sort([Point(0, 0), Point(1, 0), Point(1, 1)])
    [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]
    """
    pts = poly.vertices if isinstance(poly, Polygon) else poly
    n = len(pts)
    if n < 2:
        return list(pts)
    order = S.One if clockwise else S.NegativeOne
    dim = len(pts[0])
    if dim == 2:
        center = Point(sum((vertex.x for vertex in pts)) / n, sum((vertex.y for vertex in pts)) / n)
    else:
        center = Point(sum((vertex.x for vertex in pts)) / n, sum((vertex.y for vertex in pts)) / n, sum((vertex.z for vertex in pts)) / n)

    def compare(a, b):
        if a.x - center.x >= S.Zero and b.x - center.x < S.Zero:
            return -order
        elif a.x - center.x < 0 and b.x - center.x >= 0:
            return order
        elif a.x - center.x == 0 and b.x - center.x == 0:
            if a.y - center.y >= 0 or b.y - center.y >= 0:
                return -order if a.y > b.y else order
            return -order if b.y > a.y else order
        det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y)
        if det < 0:
            return -order
        elif det > 0:
            return order
        first = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y)
        second = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (b.y - center.y)
        return -order if first > second else order

    def compare3d(a, b):
        det = cross_product(center, a, b)
        dot_product = sum([det[i] * normal[i] for i in range(0, 3)])
        if dot_product < 0:
            return -order
        elif dot_product > 0:
            return order
    return sorted(pts, key=cmp_to_key(compare if dim == 2 else compare3d))