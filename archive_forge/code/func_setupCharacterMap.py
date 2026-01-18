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
def setupCharacterMap(self, cmapping, uvs=None, allowFallback=False):
    """Build the `cmap` table for the font. The `cmapping` argument should
        be a dict mapping unicode code points as integers to glyph names.

        The `uvs` argument, when passed, must be a list of tuples, describing
        Unicode Variation Sequences. These tuples have three elements:
            (unicodeValue, variationSelector, glyphName)
        `unicodeValue` and `variationSelector` are integer code points.
        `glyphName` may be None, to indicate this is the default variation.
        Text processors will then use the cmap to find the glyph name.
        Each Unicode Variation Sequence should be an officially supported
        sequence, but this is not policed.
        """
    subTables = []
    highestUnicode = max(cmapping) if cmapping else 0
    if highestUnicode > 65535:
        cmapping_3_1 = dict(((k, v) for k, v in cmapping.items() if k < 65536))
        subTable_3_10 = buildCmapSubTable(cmapping, 12, 3, 10)
        subTables.append(subTable_3_10)
    else:
        cmapping_3_1 = cmapping
    format = 4
    subTable_3_1 = buildCmapSubTable(cmapping_3_1, format, 3, 1)
    try:
        subTable_3_1.compile(self.font)
    except struct.error:
        if not allowFallback:
            raise ValueError('cmap format 4 subtable overflowed; sort glyph order by unicode to fix.')
        format = 12
        subTable_3_1 = buildCmapSubTable(cmapping_3_1, format, 3, 1)
    subTables.append(subTable_3_1)
    subTable_0_3 = buildCmapSubTable(cmapping_3_1, format, 0, 3)
    subTables.append(subTable_0_3)
    if uvs is not None:
        uvsDict = {}
        for unicodeValue, variationSelector, glyphName in uvs:
            if cmapping.get(unicodeValue) == glyphName:
                glyphName = None
            if variationSelector not in uvsDict:
                uvsDict[variationSelector] = []
            uvsDict[variationSelector].append((unicodeValue, glyphName))
        uvsSubTable = buildCmapSubTable({}, 14, 0, 5)
        uvsSubTable.uvsDict = uvsDict
        subTables.append(uvsSubTable)
    self.font['cmap'] = newTable('cmap')
    self.font['cmap'].tableVersion = 0
    self.font['cmap'].tables = subTables