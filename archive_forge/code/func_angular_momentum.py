from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
def angular_momentum(self, point, frame):
    """Returns the angular momentum of the rigid body about a point in the
        given frame.

        Explanation
        ===========

        The angular momentum H of a rigid body B about some point O in a frame
        N is given by:

        ``H = dot(I, w) + cross(r, M * v)``

        where I is the central inertia dyadic of B, w is the angular velocity
        of body B in the frame, N, r is the position vector from point O to the
        mass center of B, and v is the velocity of the mass center in the
        frame, N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum is desired.
        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> M, v, r, omega = dynamicsymbols('M v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, 1 * N.x)
        >>> I = outer(b.x, b.x)
        >>> B = RigidBody('B', P, b, M, (I, P))
        >>> B.angular_momentum(P, N)
        omega*b.x

        """
    I = self.central_inertia
    w = self.frame.ang_vel_in(frame)
    m = self.mass
    r = self.masscenter.pos_from(point)
    v = self.masscenter.vel(frame)
    return I.dot(w) + r.cross(m * v)