import random
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .common import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve
def rot_ccw_axis1(theta):
    """Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \\begin{bmatrix}
                1 &            0 &             0 \\\\
                0 & \\cos(\\theta) & -\\sin(\\theta) \\\\
                0 & \\sin(\\theta) &  \\cos(\\theta)
            \\end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis1(theta)
    Matrix([
    [1,         0,          0],
    [0,       1/2, -sqrt(3)/2],
    [0, sqrt(3)/2,        1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis1(pi/2)
    Matrix([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(2, 1, theta, dim=3)