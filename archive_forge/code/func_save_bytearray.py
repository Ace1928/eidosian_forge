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
def save_bytearray(self, obj):
    if self.proto < 5:
        if not obj:
            self.save_reduce(bytearray, (), obj=obj)
        else:
            self.save_reduce(bytearray, (bytes(obj),), obj=obj)
        return
    n = len(obj)
    if n >= self.framer._FRAME_SIZE_TARGET:
        self._write_large_bytes(BYTEARRAY8 + pack('<Q', n), obj)
    else:
        self.write(BYTEARRAY8 + pack('<Q', n) + obj)
    self.memoize(obj)