from fontTools.misc import sstruct
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
import pdb
import struct
def mapXMLToUTF8(string):
    uString = str()
    strLen = len(string)
    i = 0
    while i < strLen:
        prefixLen = 0
        if string[i:i + 3] == '&#x':
            prefixLen = 3
        elif string[i:i + 7] == '&amp;#x':
            prefixLen = 7
        if prefixLen:
            i = i + prefixLen
            j = i
            while string[i] != ';':
                i = i + 1
            valStr = string[j:i]
            uString = uString + chr(eval('0x' + valStr))
        else:
            uString = uString + chr(byteord(string[i]))
        i = i + 1
    return uString.encode('utf_8')