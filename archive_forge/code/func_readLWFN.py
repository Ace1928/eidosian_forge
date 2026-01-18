import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def readLWFN(path, onlyHeader=False):
    """reads an LWFN font file, returns raw data"""
    from fontTools.misc.macRes import ResourceReader
    reader = ResourceReader(path)
    try:
        data = []
        for res in reader.get('POST', []):
            code = byteord(res.data[0])
            if byteord(res.data[1]) != 0:
                raise T1Error('corrupt LWFN file')
            if code in [1, 2]:
                if onlyHeader and code == 2:
                    break
                data.append(res.data[2:])
            elif code in [3, 5]:
                break
            elif code == 4:
                with open(path, 'rb') as f:
                    data.append(f.read())
            elif code == 0:
                pass
            else:
                raise T1Error('bad chunk code: ' + repr(code))
    finally:
        reader.close()
    data = bytesjoin(data)
    assertType1(data)
    return data