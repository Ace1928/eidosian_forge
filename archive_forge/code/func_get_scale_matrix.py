from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_scale_matrix(self):
    """Copies the scaled font’s scaled matrix.

        The scale matrix is product of the font matrix
        and the ctm associated with the scaled font,
        and hence is the matrix mapping from font space to device space.

        :returns: A new :class:`Matrix` object.

        """
    matrix = Matrix()
    cairo.cairo_scaled_font_get_scale_matrix(self._pointer, matrix._pointer)
    self._check_status()
    return matrix