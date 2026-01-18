from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval
from functools import partial
from . import DefaultTable
from . import grUtils
import struct
def decompileAttributes3(self, data):
    if self.hasOctaboxes:
        o, data = sstruct.unpack2(Glat_format_3_octabox_metrics, data, _Object())
        numsub = bin(o.subboxBitmap).count('1')
        o.subboxes = []
        for b in range(numsub):
            if len(data) >= 8:
                subbox, data = sstruct.unpack2(Glat_format_3_subbox_entry, data, _Object())
                o.subboxes.append(subbox)
    attrs = self.decompileAttributes12(data, Glat_format_23_entry)
    if self.hasOctaboxes:
        attrs.octabox = o
    return attrs