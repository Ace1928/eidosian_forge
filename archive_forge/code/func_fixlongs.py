from . import DefaultTable
import struct
def fixlongs(glyphID, textLength, textOffset):
    return (int(glyphID), int(textLength), textOffset)