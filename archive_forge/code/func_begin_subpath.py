import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def begin_subpath(self, to, _open):
    """
        Start a new sub-path in the stroker.

        :param to A pointer to the start vector.

        :param _open: A boolean. If 1, the sub-path is treated as an open one.

        **Note**:

          This function is useful when you need to stroke a path that is not
          stored as an 'Outline' object.
        """
    error = FT_Stroker_BeginSubPath(self._FT_Stroker, to, _open)
    if error:
        raise FT_Exception(error)