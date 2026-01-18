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
def orient_new_quaternion(self, name, q0, q1, q2, q3, location=None, vector_names=None, variable_names=None):
    """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_quaternion('B', q0, q1, q2, q3)

        """
    orienter = QuaternionOrienter(q0, q1, q2, q3)
    return self.orient_new(name, orienter, location=location, vector_names=vector_names, variable_names=variable_names)