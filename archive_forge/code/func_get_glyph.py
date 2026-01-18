import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_glyph(self):
    """
        A function used to extract a glyph image from a slot. Note that the
        created FT_Glyph object must be released with FT_Done_Glyph.
        """
    aglyph = FT_Glyph()
    error = FT_Get_Glyph(self._FT_GlyphSlot, byref(aglyph))
    if error:
        raise FT_Exception(error)
    return Glyph(aglyph)