from fontTools.misc import sstruct
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
import pdb
import struct
def mapUTF8toXML(string):
    uString = string.decode('utf_8')
    string = ''
    for uChar in uString:
        i = ord(uChar)
        if i < 128 and i > 31:
            string = string + uChar
        else:
            string = string + '&#x' + hex(i)[2:] + ';'
    return string