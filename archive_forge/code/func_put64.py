from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def put64(self, v):
    if v < 0 or v >= 1 << 64:
        raise ProtocolBufferEncodeError('u64 too big')
    self.buf.append(v >> 0 & 255)
    self.buf.append(v >> 8 & 255)
    self.buf.append(v >> 16 & 255)
    self.buf.append(v >> 24 & 255)
    self.buf.append(v >> 32 & 255)
    self.buf.append(v >> 40 & 255)
    self.buf.append(v >> 48 & 255)
    self.buf.append(v >> 56 & 255)
    return