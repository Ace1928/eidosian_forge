import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def select_charmap(self, encoding):
    """
        Select a given charmap by its encoding tag (as listed in 'freetype.h').

        **Note**:

          This function returns an error if no charmap in the face corresponds to
          the encoding queried here.

          Because many fonts contain more than a single cmap for Unicode
          encoding, this function has some special code to select the one which
          covers Unicode best ('best' in the sense that a UCS-4 cmap is preferred
          to a UCS-2 cmap). It is thus preferable to FT_Set_Charmap in this case.
        """
    error = FT_Select_Charmap(self._FT_Face, encoding)
    if error:
        raise FT_Exception(error)