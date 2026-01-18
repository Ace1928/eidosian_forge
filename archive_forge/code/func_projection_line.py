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
def projection_line(self, line):
    """Project the given line onto the plane through the normal plane
        containing the line.

        Parameters
        ==========

        LinearEntity or LinearEntity3D

        Returns
        =======

        Point3D, Line3D, Ray3D or Segment3D

        Notes
        =====

        For the interaction between 2D and 3D lines(segments, rays), you should
        convert the line to 3D by using this method. For example for finding the
        intersection between a 2D and a 3D line, convert the 2D line to a 3D line
        by projecting it on a required plane and then proceed to find the
        intersection between those lines.

        Examples
        ========

        >>> from sympy import Plane, Line, Line3D, Point3D
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Line(Point3D(1, 1), Point3D(2, 2))
        >>> a.projection_line(b)
        Line3D(Point3D(4/3, 4/3, 1/3), Point3D(5/3, 5/3, -1/3))
        >>> c = Line3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
        >>> a.projection_line(c)
        Point3D(1, 1, 1)

        """
    if not isinstance(line, (LinearEntity, LinearEntity3D)):
        raise NotImplementedError('Enter a linear entity only')
    a, b = (self.projection(line.p1), self.projection(line.p2))
    if a == b:
        return a
    if isinstance(line, (Line, Line3D)):
        return Line3D(a, b)
    if isinstance(line, (Ray, Ray3D)):
        return Ray3D(a, b)
    if isinstance(line, (Segment, Segment3D)):
        return Segment3D(a, b)