from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def rel_curve_to(self, dx1, dy1, dx2, dy2, dx3, dy3):
    """ Relative-coordinate version of :meth:`curve_to`.
        All offsets are relative to the current point.
        Adds a cubic Bézier spline to the path from the current point
        to a point offset from the current point by ``(dx3, dy3)``,
        using points offset by ``(dx1, dy1)`` and ``(dx2, dy2)``
        as the control points.
        After this call the current point will be offset by ``(dx3, dy3)``.

        Given a current point of ``(x, y)``,
        ``context.rel_curve_to(dx1, dy1, dx2, dy2, dx3, dy3)``
        is logically equivalent to
        ``context.curve_to(x+dx1, y+dy1, x+dx2, y+dy2, x+dx3, y+dy3)``.

        :param dx1: The X offset to the first control point.
        :param dy1: The Y offset to the first control point.
        :param dx2: The X offset to the second control point.
        :param dy2: The Y offset to the second control point.
        :param dx3: The X offset to the end of the curve.
        :param dy3: The Y offset to the end of the curve.
        :type dx1: float
        :type dy1: float
        :type dx2: float
        :type dy2: float
        :type dx3: float
        :type dy3: float
        :raises:
            :exc:`CairoError` if there is no current point.
            Doing so will cause leave the context in an error state.

        """
    cairo.cairo_rel_curve_to(self._pointer, dx1, dy1, dx2, dy2, dx3, dy3)
    self._check_status()