from sympy.core.backend import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, inertia
def apply_force(self, force, point=None, reaction_body=None, reaction_point=None):
    """Add force to the body(s).

        Explanation
        ===========

        Applies the force on self or equal and oppposite forces on
        self and other body if both are given on the desried point on the bodies.
        The force applied on other body is taken opposite of self, i.e, -force.

        Parameters
        ==========

        force: Vector
            The force to be applied.
        point: Point, optional
            The point on self on which force is applied.
            By default self's masscenter.
        reaction_body: Body, optional
            Second body on which equal and opposite force
            is to be applied.
        reaction_point : Point, optional
            The point on other body on which equal and opposite
            force is applied. By default masscenter of other body.

        Example
        =======

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import Body, Point, dynamicsymbols
        >>> m, g = symbols('m g')
        >>> B = Body('B')
        >>> force1 = m*g*B.z
        >>> B.apply_force(force1) #Applying force on B's masscenter
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z)]

        We can also remove some part of force from any point on the body by
        adding the opposite force to the body on that point.

        >>> f1, f2 = dynamicsymbols('f1 f2')
        >>> P = Point('P') #Considering point P on body B
        >>> B.apply_force(f1*B.x + f2*B.y, P)
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z), (P, f1(t)*B_frame.x + f2(t)*B_frame.y)]

        Let's remove f1 from point P on body B.

        >>> B.apply_force(-f1*B.x, P)
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z), (P, f2(t)*B_frame.y)]

        To further demonstrate the use of ``apply_force`` attribute,
        consider two bodies connected through a spring.

        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> N = Body('N') #Newtonion Frame
        >>> x = dynamicsymbols('x')
        >>> B1 = Body('B1')
        >>> B2 = Body('B2')
        >>> spring_force = x*N.x

        Now let's apply equal and opposite spring force to the bodies.

        >>> P1 = Point('P1')
        >>> P2 = Point('P2')
        >>> B1.apply_force(spring_force, point=P1, reaction_body=B2, reaction_point=P2)

        We can check the loads(forces) applied to bodies now.

        >>> B1.loads
        [(P1, x(t)*N_frame.x)]
        >>> B2.loads
        [(P2, - x(t)*N_frame.x)]

        Notes
        =====

        If a new force is applied to a body on a point which already has some
        force applied on it, then the new force is added to the already applied
        force on that point.

        """
    if not isinstance(point, Point):
        if point is None:
            point = self.masscenter
        else:
            raise TypeError('Force must be applied to a point on the body.')
    if not isinstance(force, Vector):
        raise TypeError('Force must be a vector.')
    if reaction_body is not None:
        reaction_body.apply_force(-force, point=reaction_point)
    for load in self._loads:
        if point in load:
            force += load[1]
            self._loads.remove(load)
            break
    self._loads.append((point, force))