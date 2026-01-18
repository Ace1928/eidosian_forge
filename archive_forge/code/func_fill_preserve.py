from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def fill_preserve(self):
    """A drawing operator that fills the current path
        according to the current fill rule,
        (each sub-path is implicitly closed before being filled).
        Unlike :meth:`fill`,
        :meth:`fill_preserve` preserves the path within the cairo context.

        See :meth:`set_fill_rule` and :meth:`fill`.

        """
    cairo.cairo_fill_preserve(self._pointer)
    self._check_status()