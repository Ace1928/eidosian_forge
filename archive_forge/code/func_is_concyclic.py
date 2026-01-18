import warnings
from sympy.core import S, sympify, Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.numbers import Float
from sympy.core.parameters import global_parameters
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.matrices.expressions import Transpose
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable
from .entity import GeometryEntity
from mpmath.libmp.libmpf import prec_to_dps
def is_concyclic(self, *args):
    """Do `self` and the given sequence of points lie in a circle?

        Returns True if the set of points are concyclic and
        False otherwise. A trivial value of True is returned
        if there are fewer than 2 other points.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_concyclic : boolean


        Examples
        ========

        >>> from sympy import Point

        Define 4 points that are on the unit circle:

        >>> p1, p2, p3, p4 = Point(1, 0), (0, 1), (-1, 0), (0, -1)

        >>> p1.is_concyclic() == p1.is_concyclic(p2, p3, p4) == True
        True

        Define a point not on that circle:

        >>> p = Point(1, 1)

        >>> p.is_concyclic(p1, p2, p3)
        False

        """
    points = (self,) + args
    points = Point._normalize_dimension(*[Point(i) for i in points])
    points = list(uniq(points))
    if not Point.affine_rank(*points) <= 2:
        return False
    origin = points[0]
    points = [p - origin for p in points]
    mat = Matrix([list(i) + [i.dot(i)] for i in points])
    rref, pivots = mat.rref()
    if len(origin) not in pivots:
        return True
    return False