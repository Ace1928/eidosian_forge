from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_scaled_font(self):
    """Return the current scaled font.

        :return:
            A new :class:`ScaledFont` object,
            wrapping an existing cairo object.

        """
    return ScaledFont._from_pointer(cairo.cairo_get_scaled_font(self._pointer), incref=True)