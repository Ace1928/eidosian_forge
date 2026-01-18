from . import _check_status, cairo, ffi
def inverted(self):
    """Return the inverse of this matrix. See :meth:`invert`.

        :raises: :exc:`CairoError` on degenerate matrices.
        :returns: A new :class:`Matrix` object.

        """
    matrix = self.copy()
    matrix.invert()
    return matrix