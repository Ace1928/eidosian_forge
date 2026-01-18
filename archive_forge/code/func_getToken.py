from yowsup.structs import ProtocolTreeNode
import math
import binascii
import sys
import zlib
def getToken(self, index, data):
    token = self.tokenDictionary.getToken(index)
    if not token:
        index = self.readInt8(data)
        token = self.tokenDictionary.getToken(index, True)
        if not token:
            raise ValueError('Invalid token %s' % token)
    return token