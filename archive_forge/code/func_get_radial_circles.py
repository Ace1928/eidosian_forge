from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def get_radial_circles(self):
    """Return this radial gradient’s endpoint circles,
        each specified as a center coordinate and a radius.

        :returns: A ``(cx0, cy0, radius0, cx1, cy1, radius1)`` tuple of floats.

        """
    circles = ffi.new('double[6]')
    _check_status(cairo.cairo_pattern_get_radial_circles(self._pointer, circles + 0, circles + 1, circles + 2, circles + 3, circles + 4, circles + 5))
    return tuple(circles)