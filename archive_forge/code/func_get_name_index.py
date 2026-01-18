import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_name_index(self, name):
    """
        Return the glyph index of a given glyph name. This function uses driver
        specific objects to do the translation.

        :param name: The glyph name.
        """
    if not isinstance(name, bytes):
        raise FT_Exception(6, 'FT_Get_Name_Index() expects a binary string for the name parameter.')
    return FT_Get_Name_Index(self._FT_Face, name)