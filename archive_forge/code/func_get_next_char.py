import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_next_char(self, charcode, agindex):
    """
        This function is used to return the next character code in the current
        charmap of a given face following the value 'charcode', as well as the
        corresponding glyph index.

        :param charcode: The starting character code.

        :param agindex: Glyph index of next character code. 0 if charmap is empty.

        **Note**:

          You should use this function with FT_Get_First_Char to walk over all
          character codes available in a given charmap. See the note for this
          function for a simple code example.

          Note that 'agindex' is set to 0 when there are no more codes in the
          charmap.
        """
    agindex = FT_UInt(0)
    charcode = FT_Get_Next_Char(self._FT_Face, charcode, byref(agindex))
    return (charcode, agindex.value)