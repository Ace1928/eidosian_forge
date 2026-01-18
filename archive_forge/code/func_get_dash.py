from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_dash(self):
    """Return the current dash pattern.

        :returns:
            A ``(dashes, offset)`` tuple of a list and a float.
            ``dashes`` is a list of floats,
            empty if no dashing is in effect.

        """
    dashes = ffi.new('double[]', cairo.cairo_get_dash_count(self._pointer))
    offset = ffi.new('double *')
    cairo.cairo_get_dash(self._pointer, dashes, offset)
    self._check_status()
    return (list(dashes), offset[0])