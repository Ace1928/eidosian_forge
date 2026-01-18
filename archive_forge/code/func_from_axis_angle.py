from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from mpmath.libmp.libmpf import prec_to_dps
@classmethod
def from_axis_angle(cls, vector, angle):
    """Returns a rotation quaternion given the axis and the angle of rotation.

        Parameters
        ==========

        vector : tuple of three numbers
            The vector representation of the given axis.
        angle : number
            The angle by which axis is rotated (in radians).

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the given axis and the angle of rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi, sqrt
        >>> q = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), 2*pi/3)
        >>> q
        1/2 + 1/2*i + 1/2*j + 1/2*k

        """
    x, y, z = vector
    norm = sqrt(x ** 2 + y ** 2 + z ** 2)
    x, y, z = (x / norm, y / norm, z / norm)
    s = sin(angle * S.Half)
    a = cos(angle * S.Half)
    b = x * s
    c = y * s
    d = z * s
    return cls(a, b, c, d)