from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def clip_extents(self):
    """Computes a bounding box in user coordinates
        covering the area inside the current clip.

        :return:
            A ``(x1, y1, x2, y2)`` tuple of floats:
            the left, top, right and bottom of the resulting extents,
            respectively.

        """
    extents = ffi.new('double[4]')
    cairo.cairo_clip_extents(self._pointer, extents + 0, extents + 1, extents + 2, extents + 3)
    self._check_status()
    return tuple(extents)