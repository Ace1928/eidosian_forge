from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def putBoolean(self, v):
    if v:
        self.buf.append(1)
    else:
        self.buf.append(0)
    return