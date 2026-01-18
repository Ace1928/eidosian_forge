from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def locatenew(self, name, value):
    """Creates a new point with a position defined from this point.

        Parameters
        ==========

        name : str
            The name for the new point
        value : Vector
            The position of the new point relative to this point

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, Point
        >>> N = ReferenceFrame('N')
        >>> P1 = Point('P1')
        >>> P2 = P1.locatenew('P2', 10 * N.x)

        """
    if not isinstance(name, str):
        raise TypeError('Must supply a valid name')
    if value == 0:
        value = Vector(0)
    value = _check_vector(value)
    p = Point(name)
    p.set_pos(self, value)
    self.set_pos(p, -value)
    return p