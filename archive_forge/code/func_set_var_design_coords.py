import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def set_var_design_coords(self, coords, reset=False):
    """
        Set design coords. Using reset=True will set all axes to
        their default coordinates.
        """
    if reset:
        error = FT_Set_Var_Design_Coordinates(self._FT_Face, 0, 0)
    else:
        num_coords = len(coords)
        ft_coords = [int(round(c * 65536.0)) for c in coords]
        coords_array = (FT_Fixed * num_coords)(*ft_coords)
        error = FT_Set_Var_Design_Coordinates(self._FT_Face, num_coords, byref(coords_array))
    if error:
        raise FT_Exception(error)