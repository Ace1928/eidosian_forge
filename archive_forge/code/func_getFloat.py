from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getFloat(self):
    if self.idx + 4 > self.limit:
        raise ProtocolBufferDecodeError('truncated')
    a = self.buf[self.idx:self.idx + 4]
    self.idx += 4
    return struct.unpack('<f', a)[0]