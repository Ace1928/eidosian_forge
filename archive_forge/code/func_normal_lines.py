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
def normal_lines(self, p, prec=None):
    """Normal lines between `p` and the ellipse.

        Parameters
        ==========

        p : Point

        Returns
        =======

        normal_lines : list with 1, 2 or 4 Lines

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e = Ellipse((0, 0), 2, 3)
        >>> c = e.center
        >>> e.normal_lines(c + Point(1, 0))
        [Line2D(Point2D(0, 0), Point2D(1, 0))]
        >>> e.normal_lines(c)
        [Line2D(Point2D(0, 0), Point2D(0, 1)), Line2D(Point2D(0, 0), Point2D(1, 0))]

        Off-axis points require the solution of a quartic equation. This
        often leads to very large expressions that may be of little practical
        use. An approximate solution of `prec` digits can be obtained by
        passing in the desired value:

        >>> e.normal_lines((3, 3), prec=2)
        [Line2D(Point2D(-0.81, -2.7), Point2D(0.19, -1.2)),
        Line2D(Point2D(1.5, -2.0), Point2D(2.5, -2.7))]

        Whereas the above solution has an operation count of 12, the exact
        solution has an operation count of 2020.
        """
    p = Point(p, dim=2)
    if True:
        rv = []
        if p.x == self.center.x:
            rv.append(Line(self.center, slope=oo))
        if p.y == self.center.y:
            rv.append(Line(self.center, slope=0))
        if rv:
            return rv
    eq = self.equation(x, y)
    dydx = idiff(eq, y, x)
    norm = -1 / dydx
    slope = Line(p, (x, y)).slope
    seq = slope - norm
    yis = solve(seq, y)[0]
    xeq = eq.subs(y, yis).as_numer_denom()[0].expand()
    if len(xeq.free_symbols) == 1:
        try:
            xsol = Poly(xeq, x).real_roots()
        except (DomainError, PolynomialError, NotImplementedError):
            xsol = _nsort(solve(xeq, x), separated=True)[0]
        points = [Point(i, solve(eq.subs(x, i), y)[0]) for i in xsol]
    else:
        raise NotImplementedError('intersections for the general ellipse are not supported')
    slopes = [norm.subs(zip((x, y), pt.args)) for pt in points]
    if prec is not None:
        points = [pt.n(prec) for pt in points]
        slopes = [i if _not_a_coeff(i) else i.n(prec) for i in slopes]
    return [Line(pt, slope=s) for pt, s in zip(points, slopes)]