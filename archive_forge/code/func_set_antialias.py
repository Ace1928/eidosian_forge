from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def set_antialias(self, antialias):
    """Changes the :ref:`ANTIALIAS` for the font options object.
        This specifies the type of antialiasing to do when rendering text.

        """
    cairo.cairo_font_options_set_antialias(self._pointer, antialias)
    self._check_status()