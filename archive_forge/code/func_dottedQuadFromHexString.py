import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
def dottedQuadFromHexString(self, hexstr):
    return '.'.join(map(str, struct.unpack('4B', struct.pack('=L', int(hexstr, 16)))))