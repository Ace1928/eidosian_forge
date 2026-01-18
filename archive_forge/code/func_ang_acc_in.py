from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def ang_acc_in(self, otherframe):
    """Returns the angular acceleration Vector of the ReferenceFrame.

        Effectively returns the Vector:

        ``N_alpha_B``

        which represent the angular acceleration of B in N, where B is self,
        and N is otherframe.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The ReferenceFrame which the angular acceleration is returned in.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_acc(N, V)
        >>> A.ang_acc_in(N)
        10*N.x

        """
    _check_frame(otherframe)
    if otherframe in self._ang_acc_dict:
        return self._ang_acc_dict[otherframe]
    else:
        return self.ang_vel_in(otherframe).dt(otherframe)