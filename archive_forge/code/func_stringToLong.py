import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def stringToLong(s):
    if len(s) != 4:
        raise ValueError('string must be 4 bytes long')
    l = 0
    for i in range(4):
        l += byteord(s[i]) << i * 8
    return l