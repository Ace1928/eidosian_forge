import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_char_index(self, charcode):
    """
        Return the glyph index of a given character code. This function uses a
        charmap object to do the mapping.

        :param charcode: The character code.

        **Note**:

          If you use FreeType to manipulate the contents of font files directly,
          be aware that the glyph index returned by this function doesn't always
          correspond to the internal indices used within the file. This is done
          to ensure that value 0 always corresponds to the 'missing glyph'.
        """
    if isinstance(charcode, (str, unicode)):
        charcode = ord(charcode)
    return FT_Get_Char_Index(self._FT_Face, charcode)