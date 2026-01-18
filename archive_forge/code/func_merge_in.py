from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def merge_in(self, other, merge_unused=True, allowlist=None):
    entries = []
    for name, entry in other.entries.items():
        if not allowlist or name in allowlist:
            if entry.used or merge_unused:
                entries.append((name, entry))
    self.entries.update(entries)
    for attr in ('const_entries', 'type_entries', 'sue_entries', 'arg_entries', 'var_entries', 'pyfunc_entries', 'cfunc_entries', 'c_class_entries'):
        self_entries = getattr(self, attr)
        names = set((e.name for e in self_entries))
        for entry in getattr(other, attr):
            if (entry.used or merge_unused) and entry.name not in names:
                self_entries.append(entry)