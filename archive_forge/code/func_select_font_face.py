from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def select_font_face(self, family='', slant=constants.FONT_SLANT_NORMAL, weight=constants.FONT_WEIGHT_NORMAL):
    """Selects a family and style of font from a simplified description
        as a family name, slant and weight.

        .. note::

            The :meth:`select_font_face` method is part of
            what the cairo designers call the "toy" text API.
            It is convenient for short demos and simple programs,
            but it is not expected to be adequate
            for serious text-using applications.
            See :ref:`fonts` for details.

        Cairo provides no operation to list available family names
        on the system (this is a "toy", remember),
        but the standard CSS2 generic family names,
        (``"serif"``, ``"sans-serif"``, ``"cursive"``, ``"fantasy"``,
        ``"monospace"``),
        are likely to work as expected.

        If family starts with the string ``"cairo:"``,
        or if no native font backends are compiled in,
        cairo will use an internal font family.
        The internal font family recognizes many modifiers
        in the family string,
        most notably, it recognizes the string ``"monospace"``.
        That is, the family name ``"cairo:monospace"``
        will use the monospace version of the internal font family.

        If text is drawn without a call to :meth:`select_font_face`,
        (nor :meth:`set_font_face` nor :meth:`set_scaled_font`),
        the default family is platform-specific,
        but is essentially ``"sans-serif"``.
        Default slant is :obj:`NORMAL <FONT_SLANT_NORMAL>`,
        and default weight is :obj:`NORMAL <FONT_WEIGHT_NORMAL>`.

        This method is equivalent to a call to :class:`ToyFontFace`
        followed by :meth:`set_font_face`.

        """
    cairo.cairo_select_font_face(self._pointer, _encode_string(family), slant, weight)
    self._check_status()