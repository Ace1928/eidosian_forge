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
@staticmethod
def are_concurrent(*planes):
    """Is a sequence of Planes concurrent?

        Two or more Planes are concurrent if their intersections
        are a common line.

        Parameters
        ==========

        planes: list

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(5, 0, 0), normal_vector=(1, -1, 1))
        >>> b = Plane(Point3D(0, -2, 0), normal_vector=(3, 1, 1))
        >>> c = Plane(Point3D(0, -1, 0), normal_vector=(5, -1, 9))
        >>> Plane.are_concurrent(a, b)
        True
        >>> Plane.are_concurrent(a, b, c)
        False

        """
    planes = list(uniq(planes))
    for i in planes:
        if not isinstance(i, Plane):
            raise ValueError('All objects should be Planes but got %s' % i.func)
    if len(planes) < 2:
        return False
    planes = list(planes)
    first = planes.pop(0)
    sol = first.intersection(planes[0])
    if sol == []:
        return False
    else:
        line = sol[0]
        for i in planes[1:]:
            l = first.intersection(i)
            if not l or l[0] not in line:
                return False
        return True