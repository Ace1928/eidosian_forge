from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_miter_limit(self):
    """Return the current miter limit as a float."""
    return cairo.cairo_get_miter_limit(self._pointer)