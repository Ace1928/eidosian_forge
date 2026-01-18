from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval
from functools import partial
from . import DefaultTable
from . import grUtils
import struct
def compileAttributes3(self, attrs):
    if self.hasOctaboxes:
        o = attrs.octabox
        data = sstruct.pack(Glat_format_3_octabox_metrics, o)
        numsub = bin(o.subboxBitmap).count('1')
        for b in range(numsub):
            data += sstruct.pack(Glat_format_3_subbox_entry, o.subboxes[b])
    else:
        data = ''
    return data + self.compileAttributes12(attrs, Glat_format_23_entry)