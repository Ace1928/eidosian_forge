import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def set_charmap(self, charmap):
    """
        Select a given charmap for character code to glyph index mapping.

        :param charmap: A handle to the selected charmap, or an index to face->charmaps[]
        """
    if type(charmap) == Charmap:
        error = FT_Set_Charmap(self._FT_Face, charmap._FT_Charmap)
        if charmap.cmap_format == 14:
            error = 0
    else:
        error = FT_Set_Charmap(self._FT_Face, self._FT_Face.contents.charmaps[charmap])
    if error:
        raise FT_Exception(error)