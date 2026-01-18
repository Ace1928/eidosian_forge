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
def load_long(self):
    val = self.readline()[:-1]
    if val and val[-1] == b'L'[0]:
        val = val[:-1]
    self.append(int(val, 0))