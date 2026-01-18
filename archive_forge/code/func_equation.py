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
def equation(self, x=None, y=None, z=None):
    """The equation of the Plane.

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 2), Point3D(2, 4, 7), Point3D(3, 5, 1))
        >>> a.equation()
        -23*x + 11*y - 2*z + 16
        >>> a = Plane(Point3D(1, 4, 2), normal_vector=(6, 6, 6))
        >>> a.equation()
        6*x + 6*y + 6*z - 42

        """
    x, y, z = [i if i else Symbol(j, real=True) for i, j in zip((x, y, z), 'xyz')]
    a = Point3D(x, y, z)
    b = self.p1.direction_ratio(a)
    c = self.normal_vector
    return sum((i * j for i, j in zip(b, c)))