from sympy.core.backend import (S, sympify, expand, sqrt, Add, zeros, acos,
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin
from mpmath.libmp.libmpf import prec_to_dps
def magnitude(self):
    """Returns the magnitude (Euclidean norm) of self.

        Warnings
        ========

        Python ignores the leading negative sign so that might
        give wrong results.
        ``-A.x.magnitude()`` would be treated as ``-(A.x.magnitude())``,
        instead of ``(-A.x).magnitude()``.

        """
    return sqrt(self & self)