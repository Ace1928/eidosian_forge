from sympy.core.backend import (S, sympify, expand, sqrt, Add, zeros, acos,
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin
from mpmath.libmp.libmpf import prec_to_dps
def free_dynamicsymbols(self, reference_frame):
    """Returns the free dynamic symbols (functions of time ``t``) in the
        measure numbers of the vector expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free dynamic symbols of the
            given vector is to be determined.

        Returns
        =======
        set
            Set of functions of time ``t``, e.g.
            ``Function('f')(me.dynamicsymbols._t)``.

        """
    from sympy.physics.mechanics.functions import find_dynamicsymbols
    return find_dynamicsymbols(self, reference_frame=reference_frame)