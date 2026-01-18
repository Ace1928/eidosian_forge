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
@staticmethod
def rotate_point(pin, r):
    """Returns the coordinates of the point pin(a 3 tuple) after rotation.

        Parameters
        ==========

        pin : tuple
            A 3-element tuple of coordinates of a point which needs to be
            rotated.
        r : Quaternion or tuple
            Axis and angle of rotation.

            It's important to note that when r is a tuple, it must be of the form
            (axis, angle)

        Returns
        =======

        tuple
            The coordinates of the point after rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), q))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), (axis, angle)))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)

        """
    if isinstance(r, tuple):
        q = Quaternion.from_axis_angle(r[0], r[1])
    else:
        q = r.normalize()
    pout = q * Quaternion(0, pin[0], pin[1], pin[2]) * conjugate(q)
    return (pout.b, pout.c, pout.d)