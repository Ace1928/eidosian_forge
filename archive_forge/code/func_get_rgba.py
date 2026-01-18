from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def get_rgba(self):
    """Returns the solid pattern’s color.

        :returns: a ``(red, green, blue, alpha)`` tuple of floats.

        """
    rgba = ffi.new('double[4]')
    _check_status(cairo.cairo_pattern_get_rgba(self._pointer, rgba + 0, rgba + 1, rgba + 2, rgba + 3))
    return tuple(rgba)