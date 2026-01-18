from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def readInt31(self, data):
    data.pop(0)
    int1 = data.pop(0)
    int2 = data.pop(0)
    int3 = data.pop(0)
    return int1 << 24 | int1 << 16 | int2 << 8 | int3