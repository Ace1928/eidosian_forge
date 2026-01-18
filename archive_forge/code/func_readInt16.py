from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def readInt16(self, data):
    intTop = data.pop(0)
    intBot = data.pop(0)
    value = (intTop << 8) + intBot
    if value is not None:
        return value
    else:
        return ''