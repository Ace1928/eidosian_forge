from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_line_width(self):
    """Return the current line width as a float."""
    return cairo.cairo_get_line_width(self._pointer)