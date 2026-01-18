from sympy.core import Dummy, Rational, S, Symbol
from sympy.core.symbol import _symbol
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt
from .entity import GeometryEntity
from .line import (Line, Ray, Segment, Line3D, LinearEntity, LinearEntity3D,
from .point import Point, Point3D
from sympy.matrices import Matrix
from sympy.polys.polytools import cancel
from sympy.solvers import solve, linsolve
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable
from mpmath.libmp.libmpf import prec_to_dps
import random
def arbitrary_point(self, u=None, v=None):
    """ Returns an arbitrary point on the Plane. If given two
        parameters, the point ranges over the entire plane. If given 1
        or no parameters, returns a point with one parameter which,
        when varying from 0 to 2*pi, moves the point in a circle of
        radius 1 about p1 of the Plane.

        Examples
        ========

        >>> from sympy import Plane, Ray
        >>> from sympy.abc import u, v, t, r
        >>> p = Plane((1, 1, 1), normal_vector=(1, 0, 0))
        >>> p.arbitrary_point(u, v)
        Point3D(1, u + 1, v + 1)
        >>> p.arbitrary_point(t)
        Point3D(1, cos(t) + 1, sin(t) + 1)

        While arbitrary values of u and v can move the point anywhere in
        the plane, the single-parameter point can be used to construct a
        ray whose arbitrary point can be located at angle t and radius
        r from p.p1:

        >>> Ray(p.p1, _).arbitrary_point(r)
        Point3D(1, r*cos(t) + 1, r*sin(t) + 1)

        Returns
        =======

        Point3D

        """
    circle = v is None
    if circle:
        u = _symbol(u or 't', real=True)
    else:
        u = _symbol(u or 'u', real=True)
        v = _symbol(v or 'v', real=True)
    x, y, z = self.normal_vector
    a, b, c = self.p1.args
    if x.is_zero and y.is_zero:
        x1, y1, z1 = (S.One, S.Zero, S.Zero)
    else:
        x1, y1, z1 = (-y, x, S.Zero)
    x2, y2, z2 = tuple(Matrix((x, y, z)).cross(Matrix((x1, y1, z1))))
    if circle:
        x1, y1, z1 = (w / sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) for w in (x1, y1, z1))
        x2, y2, z2 = (w / sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2) for w in (x2, y2, z2))
        p = Point3D(a + x1 * cos(u) + x2 * sin(u), b + y1 * cos(u) + y2 * sin(u), c + z1 * cos(u) + z2 * sin(u))
    else:
        p = Point3D(a + x1 * u + x2 * v, b + y1 * u + y2 * v, c + z1 * u + z2 * v)
    return p