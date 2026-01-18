from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getVarInt64(self):
    result = self.getVarUint64()
    if result >= 1 << 63:
        result -= 1 << 64
    return result