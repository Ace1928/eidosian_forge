from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def set_matrix(self, matrix):
    """Sets the pattern’s transformation matrix to ``matrix``.
        This matrix is a transformation from user space to pattern space.

        When a pattern is first created
        it always has the identity matrix for its transformation matrix,
        which means that pattern space is initially identical to user space.

        **Important:**
        Please note that the direction of this transformation matrix
        is from user space to pattern space.
        This means that if you imagine the flow
        from a pattern to user space (and on to device space),
        then coordinates in that flow will be transformed
        by the inverse of the pattern matrix.

        For example, if you want to make a pattern appear twice as large
        as it does by default the correct code to use is::

            pattern.set_matrix(Matrix(xx=0.5, yy=0.5))

        Meanwhile, using values of 2 rather than 0.5 in the code above
        would cause the pattern to appear at half of its default size.

        Also, please note the discussion of the user-space locking semantics
        of :meth:`Context.set_source`.

        :param matrix: A :class:`Matrix` to be copied into the pattern.

        """
    cairo.cairo_pattern_set_matrix(self._pointer, matrix._pointer)
    self._check_status()