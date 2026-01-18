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
def load_readonly_buffer(self):
    buf = self.stack[-1]
    with memoryview(buf) as m:
        if not m.readonly:
            self.stack[-1] = m.toreadonly()