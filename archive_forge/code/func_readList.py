from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def readList(self, token, data):
    size = self.readListSize(token, data)
    listx = []
    for i in range(0, size):
        listx.append(self.nextTreeInternal(data))
    return listx