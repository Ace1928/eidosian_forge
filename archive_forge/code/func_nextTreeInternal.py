from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def nextTreeInternal(self, data):
    size = self.readListSize(self.readInt8(data), data)
    token = self.readInt8(data)
    if token == 1:
        token = self.readInt8(data)
    if token == 2:
        return None
    tag = self.readString(token, data)
    if size == 0 or tag is None:
        raise ValueError('nextTree sees 0 list or null tag')
    attribCount = (size - 2 + size % 2) / 2
    attribs = self.readAttributes(attribCount, data)
    if size % 2 == 1:
        return ProtocolTreeNode(tag, attribs)
    read2 = self.readInt8(data)
    nodeData = None
    nodeChildren = None
    if self.isListTag(read2):
        nodeChildren = self.readList(read2, data)
    elif read2 == 252:
        size = self.readInt8(data)
        nodeData = bytes(self.readArray(size, data))
    elif read2 == 253:
        size = self.readInt20(data)
        nodeData = bytes(self.readArray(size, data))
    elif read2 == 254:
        size = self.readInt31(data)
        nodeData = bytes(self.readArray(size, data))
    elif read2 in (255, 251):
        nodeData = self.readPacked8(read2, data)
    else:
        nodeData = self.readString(read2, data)
    return ProtocolTreeNode(tag, attribs, nodeChildren, nodeData)