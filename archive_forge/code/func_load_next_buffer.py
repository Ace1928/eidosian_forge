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
def load_next_buffer(self):
    if self._buffers is None:
        raise UnpicklingError('pickle stream refers to out-of-band data but no *buffers* argument was given')
    try:
        buf = next(self._buffers)
    except StopIteration:
        raise UnpicklingError('not enough out-of-band buffers')
    self.append(buf)