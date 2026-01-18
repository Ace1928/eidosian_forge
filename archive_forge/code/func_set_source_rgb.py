from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_source_rgb(self, red, green, blue):
    """Same as :meth:`set_source_rgba` with alpha always 1.
        Exists for compatibility with pycairo.

        """
    cairo.cairo_set_source_rgb(self._pointer, red, green, blue)
    self._check_status()