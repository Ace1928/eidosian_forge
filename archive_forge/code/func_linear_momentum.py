from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
def linear_momentum(self, frame):
    """ Linear momentum of the rigid body.

        Explanation
        ===========

        The linear momentum L, of a rigid body B, with respect to frame N is
        given by:

        L = M * v*

        where M is the mass of the rigid body and v* is the velocity of
        the mass center of B in the frame, N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> M, v = dynamicsymbols('M v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (N.x, N.x)
        >>> Inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, N, M, Inertia_tuple)
        >>> B.linear_momentum(N)
        M*v*N.x

        """
    return self.mass * self.masscenter.vel(frame)