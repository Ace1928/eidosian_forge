from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_weight(self):
    """Return this font face’s :ref:`FONT_WEIGHT` string."""
    return cairo.cairo_toy_font_face_get_weight(self._pointer)