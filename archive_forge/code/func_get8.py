from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def get8(self):
    if self.idx >= self.limit:
        raise ProtocolBufferDecodeError('truncated')
    c = self.buf[self.idx]
    self.idx += 1
    return c