from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_ctm(self):
    """Copies the scaled font’s font current transform matrix.

        Note that the translation offsets ``(x0, y0)`` of the CTM
        are ignored by :class:`ScaledFont`.
        So, the matrix this method returns always has 0 as ``x0`` and ``y0``.

        :returns: A new :class:`Matrix` object.

        """
    matrix = Matrix()
    cairo.cairo_scaled_font_get_ctm(self._pointer, matrix._pointer)
    self._check_status()
    return matrix