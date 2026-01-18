import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def set_pixel_sizes(self, width, height):
    """
        This function calls FT_Request_Size to request the nominal size (in
        pixels).

        :param width: The nominal width, in pixels.

        :param height: The nominal height, in pixels.
        """
    error = FT_Set_Pixel_Sizes(self._FT_Face, width, height)
    if error:
        raise FT_Exception(error)