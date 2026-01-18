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
def save_type(self, obj):
    if obj is type(None):
        return self.save_reduce(type, (None,), obj=obj)
    elif obj is type(NotImplemented):
        return self.save_reduce(type, (NotImplemented,), obj=obj)
    elif obj is type(...):
        return self.save_reduce(type, (...,), obj=obj)
    return self.save_global(obj)