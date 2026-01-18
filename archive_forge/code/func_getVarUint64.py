from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getVarUint64(self):
    result = int(0)
    shift = 0
    while 1:
        if shift >= 64:
            raise ProtocolBufferDecodeError('corrupted')
        b = self.get8()
        result |= int(b & 127) << shift
        shift += 7
        if not b & 128:
            if result >= 1 << 64:
                raise ProtocolBufferDecodeError('corrupted')
            return result
    return result