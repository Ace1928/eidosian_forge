from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def glyph_path(self, glyphs):
    """Adds closed paths for the glyphs to the current path.
        The generated path if filled,
        achieves an effect similar to that of :meth:`show_glyphs`.

        :param glyphs:
            The glyphs to show.
            See :meth:`show_text_glyphs` for the data structure.

        """
    glyphs = ffi.new('cairo_glyph_t[]', glyphs)
    cairo.cairo_glyph_path(self._pointer, glyphs, len(glyphs))
    self._check_status()