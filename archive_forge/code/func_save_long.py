from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def save_long(self, obj):
    if self.bin:
        if obj >= 0:
            if obj <= 255:
                self.write(BININT1 + pack('<B', obj))
                return
            if obj <= 65535:
                self.write(BININT2 + pack('<H', obj))
                return
        if -2147483648 <= obj <= 2147483647:
            self.write(BININT + pack('<i', obj))
            return
    if self.proto >= 2:
        encoded = encode_long(obj)
        n = len(encoded)
        if n < 256:
            self.write(LONG1 + pack('<B', n) + encoded)
        else:
            self.write(LONG4 + pack('<i', n) + encoded)
        return
    if -2147483648 <= obj <= 2147483647:
        self.write(INT + repr(obj).encode('ascii') + b'\n')
    else:
        self.write(LONG + repr(obj).encode('ascii') + b'L\n')