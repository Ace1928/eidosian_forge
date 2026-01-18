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
def parallel_plane(self, pt):
    """
        Plane parallel to the given plane and passing through the point pt.

        Parameters
        ==========

        pt: Point3D

        Returns
        =======

        Plane

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1, 4, 6), normal_vector=(2, 4, 6))
        >>> a.parallel_plane(Point3D(2, 3, 5))
        Plane(Point3D(2, 3, 5), (2, 4, 6))

        """
    a = self.normal_vector
    return Plane(pt, normal_vector=a)