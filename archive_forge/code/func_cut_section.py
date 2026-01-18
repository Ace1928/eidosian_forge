from sympy.core import Expr, S, oo, pi, sympify
from sympy.core.evalf import N
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import _symbol, Dummy, Symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from .ellipse import Circle
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray
from .point import Point
from sympy.logic import And
from sympy.matrices import Matrix
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import has_dups, has_variety, uniq, rotate_left, least_rotation
from sympy.utilities.misc import as_int, func_name
from mpmath.libmp.libmpf import prec_to_dps
import warnings
def cut_section(self, line):
    """
        Returns a tuple of two polygon segments that lie above and below
        the intersecting line respectively.

        Parameters
        ==========

        line: Line object of geometry module
            line which cuts the Polygon. The part of the Polygon that lies
            above and below this line is returned.

        Returns
        =======

        upper_polygon, lower_polygon: Polygon objects or None
            upper_polygon is the polygon that lies above the given line.
            lower_polygon is the polygon that lies below the given line.
            upper_polygon and lower polygon are ``None`` when no polygon
            exists above the line or below the line.

        Raises
        ======

        ValueError: When the line does not intersect the polygon

        Examples
        ========

        >>> from sympy import Polygon, Line
        >>> a, b = 20, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> rectangle = Polygon(p1, p2, p3, p4)
        >>> t = rectangle.cut_section(Line((0, 5), slope=0))
        >>> t
        (Polygon(Point2D(0, 10), Point2D(0, 5), Point2D(20, 5), Point2D(20, 10)),
        Polygon(Point2D(0, 5), Point2D(0, 0), Point2D(20, 0), Point2D(20, 5)))
        >>> upper_segment, lower_segment = t
        >>> upper_segment.area
        100
        >>> upper_segment.centroid
        Point2D(10, 15/2)
        >>> lower_segment.centroid
        Point2D(10, 5/2)

        References
        ==========

        .. [1] https://github.com/sympy/sympy/wiki/A-method-to-return-a-cut-section-of-any-polygon-geometry

        """
    intersection_points = self.intersection(line)
    if not intersection_points:
        raise ValueError('This line does not intersect the polygon')
    points = list(self.vertices)
    points.append(points[0])
    eq = line.equation(x, y)
    a = eq.coeff(x)
    b = eq.coeff(y)
    upper_vertices = []
    lower_vertices = []
    prev = True
    prev_point = None
    for point in points:
        compare = eq.subs({x: point.x, y: point.y}) / b if b else eq.subs(x, point.x) / a
        if compare > 0:
            if not prev:
                edge = Line(point, prev_point)
                new_point = edge.intersection(line)
                upper_vertices.append(new_point[0])
                lower_vertices.append(new_point[0])
            upper_vertices.append(point)
            prev = True
        else:
            if prev and prev_point:
                edge = Line(point, prev_point)
                new_point = edge.intersection(line)
                upper_vertices.append(new_point[0])
                lower_vertices.append(new_point[0])
            lower_vertices.append(point)
            prev = False
        prev_point = point
    upper_polygon, lower_polygon = (None, None)
    if upper_vertices and isinstance(Polygon(*upper_vertices), Polygon):
        upper_polygon = Polygon(*upper_vertices)
    if lower_vertices and isinstance(Polygon(*lower_vertices), Polygon):
        lower_polygon = Polygon(*lower_vertices)
    return (upper_polygon, lower_polygon)