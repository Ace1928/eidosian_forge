from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_font_size(self, size):
    """Sets the current font matrix to a scale by a factor of ``size``,
        replacing any font matrix previously set with :meth:`set_font_size`
        or :meth:`set_font_matrix`.
        This results in a font size of size user space units.
        (More precisely, this matrix will result in the font's
        em-square being a size by size square in user space.)

        If text is drawn without a call to :meth:`set_font_size`,
        (nor :meth:`set_font_matrix` nor :meth:`set_scaled_font`),
        the default font size is 10.0.

        :param size: The new font size, in user space units
        :type size: float

        """
    cairo.cairo_set_font_size(self._pointer, size)
    self._check_status()