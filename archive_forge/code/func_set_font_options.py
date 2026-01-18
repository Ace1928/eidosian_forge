from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_font_options(self, font_options):
    """Sets a set of custom font rendering options.
        Rendering options are derived by merging these options
        with the options derived from underlying surface;
        if the value in options has a default value
        (like :obj:`ANTIALIAS_DEFAULT`),
        then the value from the surface is used.

        :param font_options: A :class:`FontOptions` object.

        """
    cairo.cairo_set_font_options(self._pointer, font_options._pointer)
    self._check_status()