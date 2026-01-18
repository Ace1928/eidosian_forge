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
def save_frozenset(self, obj):
    save = self.save
    write = self.write
    if self.proto < 4:
        self.save_reduce(frozenset, (list(obj),), obj=obj)
        return
    write(MARK)
    for item in obj:
        save(item)
    if id(obj) in self.memo:
        write(POP_MARK + self.get(self.memo[id(obj)][0]))
        return
    write(FROZENSET)
    self.memoize(obj)