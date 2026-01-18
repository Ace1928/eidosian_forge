import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def load_glyph(self, index, flags=FT_LOAD_RENDER):
    """
        A function used to load a single glyph into the glyph slot of a face
        object.

        :param index: The index of the glyph in the font file. For CID-keyed
                      fonts (either in PS or in CFF format) this argument
                      specifies the CID value.

        :param flags: A flag indicating what to load for this glyph. The FT_LOAD_XXX
                      constants can be used to control the glyph loading process
                      (e.g., whether the outline should be scaled, whether to load
                      bitmaps or not, whether to hint the outline, etc).

        **Note**:

          The loaded glyph may be transformed. See FT_Set_Transform for the
          details.

          For subsetted CID-keyed fonts, 'FT_Err_Invalid_Argument' is returned
          for invalid CID values (this is, for CID values which don't have a
          corresponding glyph in the font). See the discussion of the
          FT_FACE_FLAG_CID_KEYED flag for more details.
        """
    error = FT_Load_Glyph(self._FT_Face, index, flags)
    if error:
        raise FT_Exception(error)