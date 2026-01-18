from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def get32(self):
    if self.idx + 4 > self.limit:
        raise ProtocolBufferDecodeError('truncated')
    c = self.buf[self.idx]
    d = self.buf[self.idx + 1]
    e = self.buf[self.idx + 2]
    f = int(self.buf[self.idx + 3])
    self.idx += 4
    return f << 24 | e << 16 | d << 8 | c