from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getDouble(self):
    if self.idx + 8 > self.limit:
        raise ProtocolBufferDecodeError('truncated')
    a = self.buf[self.idx:self.idx + 8]
    self.idx += 8
    return struct.unpack('<d', a)[0]