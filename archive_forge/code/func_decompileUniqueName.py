from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval
from . import DefaultTable
def decompileUniqueName(self, data):
    name = ''
    for char in data:
        val = byteord(char)
        if val == 0:
            break
        if val > 31 or val < 128:
            name += chr(val)
        else:
            octString = oct(val)
            if len(octString) > 3:
                octString = octString[1:]
            elif len(octString) < 3:
                octString.zfill(3)
            name += '\\' + octString
    return name