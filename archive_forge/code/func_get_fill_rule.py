from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_fill_rule(self):
    """Return the current :ref:`FILL_RULE` string."""
    return cairo.cairo_get_fill_rule(self._pointer)