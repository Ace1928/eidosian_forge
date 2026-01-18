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
def polar_second_moment_of_area(self):
    """Returns the polar second moment of area of an Ellipse

        It is a constituent of the second moment of area, linked through
        the perpendicular axis theorem. While the planar second moment of
        area describes an object's resistance to deflection (bending) when
        subjected to a force applied to a plane parallel to the central
        axis, the polar second moment of area describes an object's
        resistance to deflection when subjected to a moment applied in a
        plane perpendicular to the object's central axis (i.e. parallel to
        the cross-section)

        Examples
        ========

        >>> from sympy import symbols, Circle, Ellipse
        >>> c = Circle((5, 5), 4)
        >>> c.polar_second_moment_of_area()
        128*pi
        >>> a, b = symbols('a, b')
        >>> e = Ellipse((0, 0), a, b)
        >>> e.polar_second_moment_of_area()
        pi*a**3*b/4 + pi*a*b**3/4

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
    second_moment = self.second_moment_of_area()
    return second_moment[0] + second_moment[1]