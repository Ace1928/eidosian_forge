from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_font_matrix(self, matrix):
    """Sets the current font matrix to ``matrix``.
        The font matrix gives a transformation
        from the design space of the font
        (in this space, the em-square is 1 unit by 1 unit)
        to user space.
        Normally, a simple scale is used (see :meth:`set_font_size`),
        but a more complex font matrix can be used
        to shear the font or stretch it unequally along the two axes

        :param matrix:
            A :class:`Matrix`
            describing a transform to be applied to the current font.

        """
    cairo.cairo_set_font_matrix(self._pointer, matrix._pointer)
    self._check_status()