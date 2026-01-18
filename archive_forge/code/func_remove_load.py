from sympy.core.backend import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, inertia
def remove_load(self, about=None):
    """
        Remove load about a point or frame.

        Parameters
        ==========

        about : Point or ReferenceFrame, optional
            The point about which force is applied,
            and is to be removed.
            If about is None, then the torque about
            self's frame is removed.

        Example
        =======

        >>> from sympy.physics.mechanics import Body, Point
        >>> B = Body('B')
        >>> P = Point('P')
        >>> f1 = B.x
        >>> f2 = B.y
        >>> B.apply_force(f1)
        >>> B.apply_force(f2, P)
        >>> B.loads
        [(B_masscenter, B_frame.x), (P, B_frame.y)]

        >>> B.remove_load(P)
        >>> B.loads
        [(B_masscenter, B_frame.x)]

        """
    if about is not None:
        if not isinstance(about, Point):
            raise TypeError('Load is applied about Point or ReferenceFrame.')
    else:
        about = self.frame
    for load in self._loads:
        if about in load:
            self._loads.remove(load)
            break