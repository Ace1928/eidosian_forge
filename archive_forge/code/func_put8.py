from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def put8(self, v):
    if v < 0 or v >= 1 << 8:
        raise ProtocolBufferEncodeError('u8 too big')
    self.buf.append(v & 255)
    return