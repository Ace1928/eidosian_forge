import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def readOther(path):
    """reads any (font) file, returns raw data"""
    with open(path, 'rb') as f:
        data = f.read()
    assertType1(data)
    chunks = findEncryptedChunks(data)
    data = []
    for isEncrypted, chunk in chunks:
        if isEncrypted and isHex(chunk[:4]):
            data.append(deHexString(chunk))
        else:
            data.append(chunk)
    return bytesjoin(data)