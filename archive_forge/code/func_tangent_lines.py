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
def tangent_lines(self, p):
    """Tangent lines between `p` and the ellipse.

        If `p` is on the ellipse, returns the tangent line through point `p`.
        Otherwise, returns the tangent line(s) from `p` to the ellipse, or
        None if no tangent line is possible (e.g., `p` inside ellipse).

        Parameters
        ==========

        p : Point

        Returns
        =======

        tangent_lines : list with 1 or 2 Lines

        Raises
        ======

        NotImplementedError
            Can only find tangent lines for a point, `p`, on the ellipse.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.tangent_lines(Point(3, 0))
        [Line2D(Point2D(3, 0), Point2D(3, -12))]

        """
    p = Point(p, dim=2)
    if self.encloses_point(p):
        return []
    if p in self:
        delta = self.center - p
        rise = self.vradius ** 2 * delta.x
        run = -self.hradius ** 2 * delta.y
        p2 = Point(simplify(p.x + run), simplify(p.y + rise))
        return [Line(p, p2)]
    else:
        if len(self.foci) == 2:
            f1, f2 = self.foci
            maj = self.hradius
            test = 2 * maj - Point.distance(f1, p) - Point.distance(f2, p)
        else:
            test = self.radius - Point.distance(self.center, p)
        if test.is_number and test.is_positive:
            return []
        eq = self.equation(x, y)
        dydx = idiff(eq, y, x)
        slope = Line(p, Point(x, y)).slope
        tangent_points = solve([slope - dydx, eq], [x, y])
        if len(tangent_points) == 1:
            if tangent_points[0][0] == p.x or tangent_points[0][1] == p.y:
                return [Line(p, p + Point(1, 0)), Line(p, p + Point(0, 1))]
            else:
                return [Line(p, p + Point(0, 1)), Line(p, tangent_points[0])]
        return [Line(p, tangent_points[0]), Line(p, tangent_points[1])]