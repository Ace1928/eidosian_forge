from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def putVarInt64(self, v):
    buf_append = self.buf.append
    if v >= 9223372036854775808 or v < -9223372036854775808:
        raise ProtocolBufferEncodeError('int64 too big')
    if v < 0:
        v += 18446744073709551616
    while True:
        bits = v & 127
        v >>= 7
        if v:
            bits |= 128
        buf_append(bits)
        if not v:
            break
    return