from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def set_vel(self, frame, value):
    """Sets the velocity Vector of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which this point's velocity is defined
        value : Vector
            The vector value of this point's velocity in the frame

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_vel(N, 10 * N.x)
        >>> p1.vel(N)
        10*N.x

        """
    if value == 0:
        value = Vector(0)
    value = _check_vector(value)
    _check_frame(frame)
    self._vel_dict.update({frame: value})