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
def orient_new_body(self, name, angle1, angle2, angle3, rotation_order, location=None, vector_names=None, variable_names=None):
    """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

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
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> D = N.orient_new_body('D', q1, q2, q3, '123')

        is same as

        >>> D = N.orient_new_axis('D', q1, N.i)
        >>> D = D.orient_new_axis('D', q2, D.j)
        >>> D = D.orient_new_axis('D', q3, D.k)

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> B = N.orient_new_body('B', q1, q2, q3, '123')
        >>> B = N.orient_new_body('B', q1, q2, 0, 'ZXZ')
        >>> B = N.orient_new_body('B', 0, 0, 0, 'XYX')

        """
    orienter = BodyOrienter(angle1, angle2, angle3, rotation_order)
    return self.orient_new(name, orienter, location=location, vector_names=vector_names, variable_names=variable_names)