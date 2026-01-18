from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def font_extents(self):
    """Return the extents of the currently selected font.

        Values are given in the current user-space coordinate system.

        Because font metrics are in user-space coordinates, they are mostly,
        but not entirely, independent of the current transformation matrix.
        If you call :meth:`context.scale(2) <scale>`,
        text will be drawn twice as big,
        but the reported text extents will not be doubled.
        They will change slightly due to hinting
        (so you can't assume that metrics are independent
        of the transformation matrix),
        but otherwise will remain unchanged.

        :returns:
            A ``(ascent, descent, height, max_x_advance, max_y_advance)``
            tuple of floats.

        ``ascent``
            The distance that the font extends above the baseline.
            Note that this is not always exactly equal to
            the maximum of the extents of all the glyphs in the font,
            but rather is picked to express the font designer's intent
            as to how the font should align with elements above it.
        ``descent``
            The distance that the font extends below the baseline.
            This value is positive for typical fonts
            that include portions below the baseline.
            Note that this is not always exactly equal
            to the maximum of the extents of all the glyphs in the font,
            but rather is picked to express the font designer's intent
            as to how the font should align with elements below it.
        ``height``
            The recommended vertical distance between baselines
            when setting consecutive lines of text with the font.
            This is greater than ``ascent + descent``
            by a quantity known as the line spacing or external leading.
            When space is at a premium, most fonts can be set
            with only a distance of ``ascent + descent`` between lines.
        ``max_x_advance``
            The maximum distance in the X direction
            that the origin is advanced for any glyph in the font.
        ``max_y_advance``
            The maximum distance in the Y direction
            that the origin is advanced for any glyph in the font.
            This will be zero for normal fonts used for horizontal writing.
            (The scripts of East Asia are sometimes written vertically.)

        """
    extents = ffi.new('cairo_font_extents_t *')
    cairo.cairo_font_extents(self._pointer, extents)
    self._check_status()
    return (extents.ascent, extents.descent, extents.height, extents.max_x_advance, extents.max_y_advance)