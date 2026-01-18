from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def skipData(self, tag):
    t = tag & 7
    if t == Encoder.NUMERIC:
        self.getVarInt64()
    elif t == Encoder.DOUBLE:
        self.skip(8)
    elif t == Encoder.STRING:
        n = self.getVarInt32()
        self.skip(n)
    elif t == Encoder.STARTGROUP:
        while 1:
            t = self.getVarInt32()
            if t & 7 == Encoder.ENDGROUP:
                break
            else:
                self.skipData(t)
        if t - Encoder.ENDGROUP != tag - Encoder.STARTGROUP:
            raise ProtocolBufferDecodeError('corrupted')
    elif t == Encoder.ENDGROUP:
        raise ProtocolBufferDecodeError('corrupted')
    elif t == Encoder.FLOAT:
        self.skip(4)
    else:
        raise ProtocolBufferDecodeError('corrupted')