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
def random_point(self, seed=None):
    """ Returns a random point on the Plane.

        Returns
        =======

        Point3D

        Examples
        ========

        >>> from sympy import Plane
        >>> p = Plane((1, 0, 0), normal_vector=(0, 1, 0))
        >>> r = p.random_point(seed=42)  # seed value is optional
        >>> r.n(3)
        Point3D(2.29, 0, -1.35)

        The random point can be moved to lie on the circle of radius
        1 centered on p1:

        >>> c = p.p1 + (r - p.p1).unit
        >>> c.distance(p.p1).equals(1)
        True
        """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    params = {x: 2 * Rational(rng.gauss(0, 1)) - 1, y: 2 * Rational(rng.gauss(0, 1)) - 1}
    return self.arbitrary_point(x, y).subs(params)