from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def has_current_point(self):
    """Returns whether a current point is defined on the current path.
        See :meth:`get_current_point`.

        """
    return bool(cairo.cairo_has_current_point(self._pointer))