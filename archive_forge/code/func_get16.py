from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def get16(self):
    if self.idx + 2 > self.limit:
        raise ProtocolBufferDecodeError('truncated')
    c = self.buf[self.idx]
    d = self.buf[self.idx + 1]
    self.idx += 2
    return d << 8 | c