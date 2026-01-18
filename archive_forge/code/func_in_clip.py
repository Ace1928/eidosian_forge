from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def in_clip(self, x, y):
    """Tests whether the given point is inside the area
        that would be visible through the current clip,
        i.e. the area that would be filled by a :meth:`paint` operation.

        See :meth:`clip`, and :meth:`clip_preserve`.

        :param x: X coordinate of the point to test
        :param y: Y coordinate of the point to test
        :type x: float
        :type y: float
        :returns: A boolean.

        *New in cairo 1.10.*

        """
    return bool(cairo.cairo_in_clip(self._pointer, x, y))