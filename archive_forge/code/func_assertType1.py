import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def assertType1(data):
    for head in [b'%!PS-AdobeFont', b'%!FontType1']:
        if data[:len(head)] == head:
            break
    else:
        raise T1Error('not a PostScript font')
    if not _fontType1RE.search(data):
        raise T1Error('not a Type 1 font')
    if data.find(b'currentfile eexec') < 0:
        raise T1Error('not an encrypted Type 1 font')
    return data