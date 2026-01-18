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
def load_long_binget(self):
    i, = unpack('<I', self.read(4))
    try:
        self.append(self.memo[i])
    except KeyError as exc:
        msg = f'Memo value not found at index {i}'
        raise UnpicklingError(msg) from None