from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def get64(self):
    if self.idx + 8 > self.limit:
        raise ProtocolBufferDecodeError('truncated')
    c = self.buf[self.idx]
    d = self.buf[self.idx + 1]
    e = self.buf[self.idx + 2]
    f = int(self.buf[self.idx + 3])
    g = int(self.buf[self.idx + 4])
    h = int(self.buf[self.idx + 5])
    i = int(self.buf[self.idx + 6])
    j = int(self.buf[self.idx + 7])
    self.idx += 8
    return j << 56 | i << 48 | h << 40 | g << 32 | f << 24 | e << 16 | d << 8 | c