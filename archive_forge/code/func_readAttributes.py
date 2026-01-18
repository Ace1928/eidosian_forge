from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def readAttributes(self, attribCount, data):
    attribs = {}
    for i in range(0, int(attribCount)):
        key = self.readString(self.readInt8(data), data)
        value = self.readString(self.readInt8(data), data)
        attribs[key] = value
    return attribs