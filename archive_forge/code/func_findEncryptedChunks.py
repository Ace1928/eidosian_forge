import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def findEncryptedChunks(data):
    chunks = []
    while True:
        eBegin = data.find(EEXECBEGIN)
        if eBegin < 0:
            break
        eBegin = eBegin + len(EEXECBEGIN) + 1
        endMatch = EEXECEND.search(data, eBegin)
        if endMatch is None:
            raise T1Error("can't find end of eexec part")
        eEnd = endMatch.start()
        cypherText = data[eBegin:eEnd + 2]
        if isHex(cypherText[:4]):
            cypherText = deHexString(cypherText)
        plainText, R = eexec.decrypt(cypherText, 55665)
        eEndLocal = plainText.find(EEXECINTERNALEND)
        if eEndLocal < 0:
            raise T1Error("can't find end of eexec part")
        chunks.append((0, data[:eBegin]))
        chunks.append((1, cypherText[:eEndLocal + len(EEXECINTERNALEND) + 1]))
        data = data[eEnd:]
    chunks.append((0, data))
    return chunks