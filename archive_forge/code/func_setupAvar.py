from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from .ttLib import TTFont, newTable
from .ttLib.tables._c_m_a_p import cmap_classes
from .ttLib.tables._g_l_y_f import flagCubic
from .ttLib.tables.O_S_2f_2 import Panose
from .misc.timeTools import timestampNow
import struct
from collections import OrderedDict
def setupAvar(self, axes, mappings=None):
    """Adds an axis variations table to the font.

        Args:
            axes (list): A list of py:class:`.designspaceLib.AxisDescriptor` objects.
        """
    from .varLib import _add_avar
    if 'fvar' not in self.font:
        raise KeyError("'fvar' table is missing; can't add 'avar'.")
    axisTags = [axis.axisTag for axis in self.font['fvar'].axes]
    axes = OrderedDict(enumerate(axes))
    _add_avar(self.font, axes, mappings, axisTags)