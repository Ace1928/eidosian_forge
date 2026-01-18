from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def putVarUint64(self, v):
    buf_append = self.buf.append
    if v < 0 or v >= 18446744073709551616:
        raise ProtocolBufferEncodeError('uint64 too big')
    while True:
        bits = v & 127
        v >>= 7
        if v:
            bits |= 128
        buf_append(bits)
        if not v:
            break
    return