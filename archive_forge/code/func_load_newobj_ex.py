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
def load_newobj_ex(self):
    kwargs = self.stack.pop()
    args = self.stack.pop()
    cls = self.stack.pop()
    obj = cls.__new__(cls, *args, **kwargs)
    self.append(obj)