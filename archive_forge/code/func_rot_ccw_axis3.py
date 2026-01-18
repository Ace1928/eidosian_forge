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
def rot_ccw_axis3(theta):
    """Returns a rotation matrix for a rotation of theta (in radians)
    about the 3-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `z`-axis, given by:

    .. math::

        R  = \\begin{bmatrix}
                \\cos(\\theta) & -\\sin(\\theta) & 0 \\\\
                \\sin(\\theta) &  \\cos(\\theta) & 0 \\\\
                           0 &             0 & 1
            \\end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis3

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis3(theta)
    Matrix([
    [      1/2, -sqrt(3)/2, 0],
    [sqrt(3)/2,        1/2, 0],
    [        0,          0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis3(pi/2)
    Matrix([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    """
    return rot_givens(1, 0, theta, dim=3)