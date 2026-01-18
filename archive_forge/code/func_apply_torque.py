from sympy.core.backend import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, inertia
def apply_torque(self, torque, reaction_body=None):
    """Add torque to the body(s).

        Explanation
        ===========

        Applies the torque on self or equal and oppposite torquess on
        self and other body if both are given.
        The torque applied on other body is taken opposite of self,
        i.e, -torque.

        Parameters
        ==========

        torque: Vector
            The torque to be applied.
        reaction_body: Body, optional
            Second body on which equal and opposite torque
            is to be applied.

        Example
        =======

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> t = symbols('t')
        >>> B = Body('B')
        >>> torque1 = t*B.z
        >>> B.apply_torque(torque1)
        >>> B.loads
        [(B_frame, t*B_frame.z)]

        We can also remove some part of torque from the body by
        adding the opposite torque to the body.

        >>> t1, t2 = dynamicsymbols('t1 t2')
        >>> B.apply_torque(t1*B.x + t2*B.y)
        >>> B.loads
        [(B_frame, t1(t)*B_frame.x + t2(t)*B_frame.y + t*B_frame.z)]

        Let's remove t1 from Body B.

        >>> B.apply_torque(-t1*B.x)
        >>> B.loads
        [(B_frame, t2(t)*B_frame.y + t*B_frame.z)]

        To further demonstrate the use, let us consider two bodies such that
        a torque `T` is acting on one body, and `-T` on the other.

        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> N = Body('N') #Newtonion frame
        >>> B1 = Body('B1')
        >>> B2 = Body('B2')
        >>> v = dynamicsymbols('v')
        >>> T = v*N.y #Torque

        Now let's apply equal and opposite torque to the bodies.

        >>> B1.apply_torque(T, B2)

        We can check the loads (torques) applied to bodies now.

        >>> B1.loads
        [(B1_frame, v(t)*N_frame.y)]
        >>> B2.loads
        [(B2_frame, - v(t)*N_frame.y)]

        Notes
        =====

        If a new torque is applied on body which already has some torque applied on it,
        then the new torque is added to the previous torque about the body's frame.

        """
    if not isinstance(torque, Vector):
        raise TypeError('A Vector must be supplied to add torque.')
    if reaction_body is not None:
        reaction_body.apply_torque(-torque)
    for load in self._loads:
        if self.frame in load:
            torque += load[1]
            self._loads.remove(load)
            break
    self._loads.append((self.frame, torque))