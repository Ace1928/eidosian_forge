from sympy.core import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, symbols
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point, Point2D
from sympy.geometry.line import Line, Line2D, Ray2D, Segment2D, LinearEntity3D
from sympy.geometry.ellipse import Ellipse
from sympy.functions import sign
from sympy.simplify import simplify
from sympy.solvers.solvers import solve
@property
def focal_length(self):
    """The focal length of the parabola.

        Returns
        =======

        focal_lenght : number or symbolic expression

        Notes
        =====

        The distance between the vertex and the focus
        (or the vertex and directrix), measured along the axis
        of symmetry, is the "focal length".

        See Also
        ========

        https://en.wikipedia.org/wiki/Parabola

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.focal_length
        4

        """
    distance = self.directrix.distance(self.focus)
    focal_length = distance / 2
    return focal_length