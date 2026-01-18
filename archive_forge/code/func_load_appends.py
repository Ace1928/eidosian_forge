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
def load_appends(self):
    items = self.pop_mark()
    list_obj = self.stack[-1]
    try:
        extend = list_obj.extend
    except AttributeError:
        pass
    else:
        extend(items)
        return
    append = list_obj.append
    for item in items:
        append(item)