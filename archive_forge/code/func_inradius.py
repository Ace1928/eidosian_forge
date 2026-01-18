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
def inradius(self):
    """The radius of the incircle.

        Returns
        =======

        inradius : number of Basic instance

        See Also
        ========

        incircle, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(4, 0), Point(0, 3)
        >>> t = Triangle(p1, p2, p3)
        >>> t.inradius
        1

        """
    return simplify(2 * self.area / self.perimeter)