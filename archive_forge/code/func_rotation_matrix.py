from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy.core.symbol import Str
from sympy.core.symbol import symbols
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrices import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
from sympy.vector.vector import BaseVector
def rotation_matrix(self, other):
    """
        Returns the direction cosine matrix(DCM), also known as the
        'rotation matrix' of this coordinate system with respect to
        another system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.rotation_matrix(B) * v_b.

        A SymPy Matrix is returned.

        Parameters
        ==========

        other : CoordSys3D
            The system which the DCM is generated to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> A = N.orient_new_axis('A', q1, N.i)
        >>> N.rotation_matrix(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        """
    from sympy.vector.functions import _path
    if not isinstance(other, CoordSys3D):
        raise TypeError(str(other) + ' is not a CoordSys3D')
    if other == self:
        return eye(3)
    elif other == self._parent:
        return self._parent_rotation_matrix
    elif other._parent == self:
        return other._parent_rotation_matrix.T
    rootindex, path = _path(self, other)
    result = eye(3)
    i = -1
    for i in range(rootindex):
        result *= path[i]._parent_rotation_matrix
    i += 2
    while i < len(path):
        result *= path[i]._parent_rotation_matrix.T
        i += 1
    return result