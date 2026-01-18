import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def readPFB(path, onlyHeader=False):
    """reads a PFB font file, returns raw data"""
    data = []
    with open(path, 'rb') as f:
        while True:
            if f.read(1) != bytechr(128):
                raise T1Error('corrupt PFB file')
            code = byteord(f.read(1))
            if code in [1, 2]:
                chunklen = stringToLong(f.read(4))
                chunk = f.read(chunklen)
                assert len(chunk) == chunklen
                data.append(chunk)
            elif code == 3:
                break
            else:
                raise T1Error('bad chunk code: ' + repr(code))
            if onlyHeader:
                break
    data = bytesjoin(data)
    assertType1(data)
    return data