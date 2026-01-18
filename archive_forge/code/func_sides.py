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
@property
def sides(self):
    """The directed line segments that form the sides of the polygon.

        Returns
        =======

        sides : list of sides
            Each side is a directed Segment.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.sides
        [Segment2D(Point2D(0, 0), Point2D(1, 0)),
        Segment2D(Point2D(1, 0), Point2D(5, 1)),
        Segment2D(Point2D(5, 1), Point2D(0, 1)), Segment2D(Point2D(0, 1), Point2D(0, 0))]

        """
    res = []
    args = self.vertices
    for i in range(-len(args), 0):
        res.append(Segment(args[i], args[i + 1]))
    return res