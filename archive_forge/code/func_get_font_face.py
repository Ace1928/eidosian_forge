from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_font_face(self):
    """Return the font face that this scaled font uses.

        :returns:
            A new instance of :class:`FontFace` (or one of its sub-classes).
            Might wrap be the same font face passed to :class:`ScaledFont`,
            but this does not hold true for all possible cases.

        """
    return FontFace._from_pointer(cairo.cairo_scaled_font_get_font_face(self._pointer), incref=True)