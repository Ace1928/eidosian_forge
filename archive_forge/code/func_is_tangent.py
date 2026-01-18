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
def is_tangent(self, o):
    """Is `o` tangent to the ellipse?

        Parameters
        ==========

        o : GeometryEntity
            An Ellipse, LinearEntity or Polygon

        Raises
        ======

        NotImplementedError
            When the wrong type of argument is supplied.

        Returns
        =======

        is_tangent: boolean
            True if o is tangent to the ellipse, False otherwise.

        See Also
        ========

        tangent_lines

        Examples
        ========

        >>> from sympy import Point, Ellipse, Line
        >>> p0, p1, p2 = Point(0, 0), Point(3, 0), Point(3, 3)
        >>> e1 = Ellipse(p0, 3, 2)
        >>> l1 = Line(p1, p2)
        >>> e1.is_tangent(l1)
        True

        """
    if isinstance(o, Point2D):
        return False
    elif isinstance(o, Ellipse):
        intersect = self.intersection(o)
        if isinstance(intersect, Ellipse):
            return True
        elif intersect:
            return all((self.tangent_lines(i)[0].equals(o.tangent_lines(i)[0]) for i in intersect))
        else:
            return False
    elif isinstance(o, Line2D):
        hit = self.intersection(o)
        if not hit:
            return False
        if len(hit) == 1:
            return True
        return hit[0].equals(hit[1])
    elif isinstance(o, Ray2D):
        intersect = self.intersection(o)
        if len(intersect) == 1:
            return intersect[0] != o.source and (not self.encloses_point(o.source))
        else:
            return False
    elif isinstance(o, (Segment2D, Polygon)):
        all_tangents = False
        segments = o.sides if isinstance(o, Polygon) else [o]
        for segment in segments:
            intersect = self.intersection(segment)
            if len(intersect) == 1:
                if not any((intersect[0] in i for i in segment.points)) and (not any((self.encloses_point(i) for i in segment.points))):
                    all_tangents = True
                    continue
                else:
                    return False
            else:
                return all_tangents
        return all_tangents
    elif isinstance(o, (LinearEntity3D, Point3D)):
        raise TypeError('Entity must be two dimensional, not three dimensional')
    else:
        raise TypeError('Is_tangent not handled for %s' % func_name(o))