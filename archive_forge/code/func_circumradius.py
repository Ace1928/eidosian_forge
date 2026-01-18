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
def circumradius(self):
    """The radius of the circumcircle of the triangle.

        Returns
        =======

        circumradius : number of Basic instance

        See Also
        ========

        sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import Point, Triangle
        >>> a = Symbol('a')
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, a)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumradius
        sqrt(a**2/4 + 1/4)
        """
    return Point.distance(self.circumcenter, self.vertices[0])