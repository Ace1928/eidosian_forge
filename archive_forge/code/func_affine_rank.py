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
@staticmethod
def affine_rank(*args):
    """The affine rank of a set of points is the dimension
        of the smallest affine space containing all the points.
        For example, if the points lie on a line (and are not all
        the same) their affine rank is 1.  If the points lie on a plane
        but not a line, their affine rank is 2.  By convention, the empty
        set has affine rank -1."""
    if len(args) == 0:
        return -1
    points = Point._normalize_dimension(*[Point(i) for i in args])
    origin = points[0]
    points = [i - origin for i in points[1:]]
    m = Matrix([i.args for i in points])
    return m.rank(iszerofunc=lambda x: abs(x.n(2)) < 1e-12 if x.is_number else x.is_zero)