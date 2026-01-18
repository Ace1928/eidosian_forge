from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def in_fill(self, x, y):
    """Tests whether the given point is inside the area
        that would be affected by a :meth:`fill` operation
        given the current path and filling parameters.
        Surface dimensions and clipping are not taken into account.

        See :meth:`fill`, :meth:`set_fill_rule` and :meth:`fill_preserve`.

        :param x: X coordinate of the point to test
        :param y: Y coordinate of the point to test
        :type x: float
        :type y: float
        :returns: A boolean.

        """
    return bool(cairo.cairo_in_fill(self._pointer, x, y))