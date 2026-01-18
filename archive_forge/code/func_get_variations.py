from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_variations(self):
    """Gets the OpenType font variations for the font options object.

        See :meth:`set_variations` for details about the
        string format.

        :return: the font variations for the font options object. The
          returned string belongs to the ``options`` and must not be modified.
          It is valid until either the font options object is destroyed or the
          font variations in this object is modified with
          :meth:`set_variations`.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
    variations = cairo.cairo_font_options_get_variations(self._pointer)
    if variations != ffi.NULL:
        return ffi.string(variations).decode('utf8', 'replace')