from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def left_integral2D(m, index, facets, x0, expr, gens):
    """Computes the left integral of Eq 10 in Chin et al.
    For the 2D case, the integral is just an evaluation of the polynomial
    at the intersection of two facets which is multiplied by the distance
    between the first point of facet and that intersection.

    Parameters
    ==========

    m :
        No. of hyperplanes.
    index :
        Index of facet to find intersections with.
    facets :
        List of facets(Line Segments in 2D case).
    x0 :
        First point on facet referenced by index.
    expr :
        Input polynomial
    gens :
        Generators which generate the polynomial

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import left_integral2D
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> left_integral2D(3, 0, facets, facets[0].points[0], 1, (x, y))
    5
    """
    value = S.Zero
    for j in range(m):
        intersect = ()
        if j in ((index - 1) % m, (index + 1) % m):
            intersect = intersection(facets[index], facets[j], 'segment2D')
        if intersect:
            distance_origin = norm(tuple(map(lambda x, y: x - y, intersect, x0)))
            if is_vertex(intersect):
                if isinstance(expr, Expr):
                    if len(gens) == 3:
                        expr_dict = {gens[0]: intersect[0], gens[1]: intersect[1], gens[2]: intersect[2]}
                    else:
                        expr_dict = {gens[0]: intersect[0], gens[1]: intersect[1]}
                    value += distance_origin * expr.subs(expr_dict)
                else:
                    value += distance_origin * expr
    return value