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
def perpendicular_plane(self, *pts):
    """
        Return a perpendicular passing through the given points. If the
        direction ratio between the points is the same as the Plane's normal
        vector then, to select from the infinite number of possible planes,
        a third point will be chosen on the z-axis (or the y-axis
        if the normal vector is already parallel to the z-axis). If less than
        two points are given they will be supplied as follows: if no point is
        given then pt1 will be self.p1; if a second point is not given it will
        be a point through pt1 on a line parallel to the z-axis (if the normal
        is not already the z-axis, otherwise on the line parallel to the
        y-axis).

        Parameters
        ==========

        pts: 0, 1 or 2 Point3D

        Returns
        =======

        Plane

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a, b = Point3D(0, 0, 0), Point3D(0, 1, 0)
        >>> Z = (0, 0, 1)
        >>> p = Plane(a, normal_vector=Z)
        >>> p.perpendicular_plane(a, b)
        Plane(Point3D(0, 0, 0), (1, 0, 0))
        """
    if len(pts) > 2:
        raise ValueError('No more than 2 pts should be provided.')
    pts = list(pts)
    if len(pts) == 0:
        pts.append(self.p1)
    if len(pts) == 1:
        x, y, z = self.normal_vector
        if x == y == 0:
            dir = (0, 1, 0)
        else:
            dir = (0, 0, 1)
        pts.append(pts[0] + Point3D(*dir))
    p1, p2 = [Point(i, dim=3) for i in pts]
    l = Line3D(p1, p2)
    n = Line3D(p1, direction_ratio=self.normal_vector)
    if l in n:
        x, y, z = self.normal_vector
        if x == y == 0:
            p3 = Point3D(0, 1, 0)
        else:
            p3 = Point3D(0, 0, 1)
        if p3 in l:
            p3 *= 2
    else:
        p3 = p1 + Point3D(*self.normal_vector)
    return Plane(p1, p2, p3)