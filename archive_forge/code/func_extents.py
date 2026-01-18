from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def extents(self):
    """Return the scaled font’s extents.
        See :meth:`Context.font_extents`.

        :returns:
            A ``(ascent, descent, height, max_x_advance, max_y_advance)``
            tuple of floats.

        """
    extents = ffi.new('cairo_font_extents_t *')
    cairo.cairo_scaled_font_extents(self._pointer, extents)
    self._check_status()
    return (extents.ascent, extents.descent, extents.height, extents.max_x_advance, extents.max_y_advance)