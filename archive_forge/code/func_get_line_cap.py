from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_line_cap(self):
    """Return the current :ref:`LINE_CAP` string."""
    return cairo.cairo_get_line_cap(self._pointer)