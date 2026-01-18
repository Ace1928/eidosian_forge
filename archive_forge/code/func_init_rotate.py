from . import _check_status, cairo, ffi
@classmethod
def init_rotate(cls, radians):
    """Return a new :class:`Matrix` for a transformation
        that rotates by ``radians``.

        :type radians: float
        :param radians:
            Angle of rotation, in radians.
            The direction of rotation is defined such that
            positive angles rotate in the direction
            from the positive X axis toward the positive Y axis.
            With the default axis orientation of cairo,
            positive angles rotate in a clockwise direction.

        """
    result = cls()
    cairo.cairo_matrix_init_rotate(result._pointer, radians)
    return result