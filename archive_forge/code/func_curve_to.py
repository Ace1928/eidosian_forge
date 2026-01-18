from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def curve_to(self, x1, y1, x2, y2, x3, y3):
    """Adds a cubic Bézier spline to the path
        from the current point
        to position ``(x3, y3)`` in user-space coordinates,
        using ``(x1, y1)`` and ``(x2, y2)`` as the control points.
        After this call the current point will be ``(x3, y3)``.

        If there is no current point before the call to :meth:`curve_to`
        this method will behave as if preceded by
        a call to ``context.move_to(x1, y1)``.

        :param x1: The X coordinate of the first control point.
        :param y1: The Y coordinate of the first control point.
        :param x2: The X coordinate of the second control point.
        :param y2: The Y coordinate of the second control point.
        :param x3: The X coordinate of the end of the curve.
        :param y3: The Y coordinate of the end of the curve.
        :type x1: float
        :type y1: float
        :type x2: float
        :type y2: float
        :type x3: float
        :type y3: float

        """
    cairo.cairo_curve_to(self._pointer, x1, y1, x2, y2, x3, y3)
    self._check_status()