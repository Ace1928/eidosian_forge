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
@property
def orthogonal_direction(self):
    """Returns a non-zero point that is orthogonal to the
        line containing `self` and the origin.

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2, 3)
        >>> a.orthogonal_direction
        Point3D(-2, 1, 0)
        >>> b = _
        >>> Line(b, b.origin).is_perpendicular(Line(a, a.origin))
        True
        """
    dim = self.ambient_dimension
    if self[0].is_zero:
        return Point([1] + (dim - 1) * [0])
    if self[1].is_zero:
        return Point([0, 1] + (dim - 2) * [0])
    return Point([-self[1], self[0]] + (dim - 2) * [0])