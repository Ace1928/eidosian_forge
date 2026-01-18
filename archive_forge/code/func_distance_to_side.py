from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def distance_to_side(point, line_seg, A):
    """Helper function to compute the signed distance between given 3D point
    and a line segment.

    Parameters
    ==========

    point : 3D Point
    line_seg : Line Segment

    Examples
    ========

    >>> from sympy.integrals.intpoly import distance_to_side
    >>> point = (0, 0, 0)
    >>> distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0))
    -sqrt(2)/2
    """
    x1, x2 = line_seg
    rev_normal = [-1 * S(i) / norm(A) for i in A]
    vector = [x2[i] - x1[i] for i in range(0, 3)]
    vector = [vector[i] / norm(vector) for i in range(0, 3)]
    n_side = cross_product((0, 0, 0), rev_normal, vector)
    vectorx0 = [line_seg[0][i] - point[i] for i in range(0, 3)]
    dot_product = sum([vectorx0[i] * n_side[i] for i in range(0, 3)])
    return dot_product