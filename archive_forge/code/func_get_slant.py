from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_slant(self):
    """Return this font face’s :ref:`FONT_SLANT` string."""
    return cairo.cairo_toy_font_face_get_slant(self._pointer)