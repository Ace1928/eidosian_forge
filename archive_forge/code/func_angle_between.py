from sympy.core.backend import (S, sympify, expand, sqrt, Add, zeros, acos,
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin
from mpmath.libmp.libmpf import prec_to_dps
def angle_between(self, vec):
    """
        Returns the smallest angle between Vector 'vec' and self.

        Parameter
        =========

        vec : Vector
            The Vector between which angle is needed.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame("A")
        >>> v1 = A.x
        >>> v2 = A.y
        >>> v1.angle_between(v2)
        pi/2

        >>> v3 = A.x + A.y + A.z
        >>> v1.angle_between(v3)
        acos(sqrt(3)/3)

        Warnings
        ========

        Python ignores the leading negative sign so that might give wrong
        results. ``-A.x.angle_between()`` would be treated as
        ``-(A.x.angle_between())``, instead of ``(-A.x).angle_between()``.

        """
    vec1 = self.normalize()
    vec2 = vec.normalize()
    angle = acos(vec1.dot(vec2))
    return angle