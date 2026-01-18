from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getVarInt32(self):
    b = self.get8()
    if not b & 128:
        return b
    result = int(0)
    shift = 0
    while 1:
        result |= int(b & 127) << shift
        shift += 7
        if not b & 128:
            if result >= 18446744073709551616:
                raise ProtocolBufferDecodeError('corrupted')
            break
        if shift >= 64:
            raise ProtocolBufferDecodeError('corrupted')
        b = self.get8()
    if result >= 9223372036854775808:
        result -= 18446744073709551616
    if result >= 2147483648 or result < -2147483648:
        raise ProtocolBufferDecodeError('corrupted')
    return result