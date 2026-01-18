from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def stroke_preserve(self):
    """A drawing operator that strokes the current path
        according to the current line width, line join, line cap,
        and dash settings.
        Unlike :meth:`stroke`,
        :meth:`stroke_preserve` preserves the path within the cairo context.
        See :meth:`set_line_width`, :meth:`set_line_join`,
        :meth:`set_line_cap`, :meth:`set_dash`, and :meth:`stroke`.

        """
    cairo.cairo_stroke_preserve(self._pointer)
    self._check_status()