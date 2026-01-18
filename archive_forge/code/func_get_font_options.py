from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_font_options(self):
    """Copies the scaled font’s options.

        :returns: A new :class:`FontOptions` object.

        """
    font_options = FontOptions()
    cairo.cairo_scaled_font_get_font_options(self._pointer, font_options._pointer)
    return font_options