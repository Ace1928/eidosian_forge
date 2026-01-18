from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_dash_count(self):
    """Same as ``len(context.get_dash()[0])``."""
    return cairo.cairo_get_dash_count(self._pointer)