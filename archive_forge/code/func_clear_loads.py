from sympy.core.backend import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, inertia
def clear_loads(self):
    """
        Clears the Body's loads list.

        Example
        =======

        >>> from sympy.physics.mechanics import Body
        >>> B = Body('B')
        >>> force = B.x + B.y
        >>> B.apply_force(force)
        >>> B.loads
        [(B_masscenter, B_frame.x + B_frame.y)]
        >>> B.clear_loads()
        >>> B.loads
        []

        """
    self._loads = []