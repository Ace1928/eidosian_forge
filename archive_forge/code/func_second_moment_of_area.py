from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core import S, pi, sympify
from sympy.core.evalf import N
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, uniquely_named_symbol, _symbol
from sympy.simplify import simplify, trigsimp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray2D, Segment2D, Line2D, LinearEntity3D
from .point import Point, Point2D, Point3D
from .util import idiff, find
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name
from mpmath.libmp.libmpf import prec_to_dps
import random
from .polygon import Polygon, Triangle
def second_moment_of_area(self, point=None):
    """Returns the second moment and product moment area of an ellipse.

        Parameters
        ==========

        point : Point, two-tuple of sympifiable objects, or None(default=None)
            point is the point about which second moment of area is to be found.
            If "point=None" it will be calculated about the axis passing through the
            centroid of the ellipse.

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
            I_xx, I_yy are second moment of area of an ellise.
            I_xy is product moment of area of an ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.second_moment_of_area()
        (3*pi/4, 27*pi/4, 0)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/List_of_second_moments_of_area

        """
    I_xx = S.Pi * self.hradius * self.vradius ** 3 / 4
    I_yy = S.Pi * self.hradius ** 3 * self.vradius / 4
    I_xy = 0
    if point is None:
        return (I_xx, I_yy, I_xy)
    I_xx = I_xx + self.area * (point[1] - self.center.y) ** 2
    I_yy = I_yy + self.area * (point[0] - self.center.x) ** 2
    I_xy = I_xy + self.area * (point[0] - self.center.x) * (point[1] - self.center.y)
    return (I_xx, I_yy, I_xy)