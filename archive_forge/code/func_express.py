from sympy.core.backend import (S, sympify, expand, sqrt, Add, zeros, acos,
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin
from mpmath.libmp.libmpf import prec_to_dps
def express(self, otherframe, variables=False):
    """
        Returns a Vector equivalent to this one, expressed in otherframe.
        Uses the global express method.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame for this Vector to be described in

        variables : boolean
            If True, the coordinate symbols(if present) in this Vector
            are re-expressed in terms otherframe

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.express(N)
        cos(q1)*N.x - sin(q1)*N.z

        """
    from sympy.physics.vector import express
    return express(self, otherframe, variables=variables)