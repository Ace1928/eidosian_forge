import os
import ctypes
import platform
import subprocess
import collections
import math
import freetype
from freetype.raw import FT_Outline_Get_Bitmap, FT_Outline_Get_BBox, FT_Outline_Get_CBox
from freetype.ft_types import FT_Pos
from freetype.ft_structs import FT_Vector, FT_BBox, FT_Bitmap, FT_Outline
from freetype.ft_enums import (
from freetype.ft_errors import FT_Exception
from fontTools.pens.basePen import BasePen, PenError
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform
@property
def cbox(self):
    """Returns an outline's ‘control box’.

        Returns:
            A tuple of ``(xMin, yMin, xMax, yMax)``.
        """
    cbox = FT_BBox()
    outline = self.outline()
    FT_Outline_Get_CBox(ctypes.byref(outline), ctypes.byref(cbox))
    return (cbox.xMin / 64.0, cbox.yMin / 64.0, cbox.xMax / 64.0, cbox.yMax / 64.0)