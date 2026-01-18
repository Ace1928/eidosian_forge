from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def getProtocolTreeNode(self, data):
    if type(data) is list:
        data = bytearray(data)
    if data[0] & self.tokenDictionary.FLAG_DEFLATE != 0:
        data = bytearray(b'\x00' + zlib.decompress(bytes(data[1:])))
    if data[0] & self.tokenDictionary.FLAG_SEGMENTED != 0:
        raise ValueError('server to client stanza fragmentation not supported')
    return self.nextTreeInternal(data[1:])