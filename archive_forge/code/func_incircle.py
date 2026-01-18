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
def incircle(self):
    """The incircle of the triangle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(2, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incircle
        Circle(Point2D(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))

        """
    return Circle(self.incenter, self.inradius)