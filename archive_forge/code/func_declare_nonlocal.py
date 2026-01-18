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
def declare_nonlocal(self, name, pos):
    orig_entry = self.lookup_here(name)
    if orig_entry and orig_entry.scope is self and (not orig_entry.from_closure):
        error(pos, "'%s' redeclared as nonlocal" % name)
        orig_entry.already_declared_here()
    else:
        entry = self.lookup(name)
        if entry is None:
            error(pos, "no binding for nonlocal '%s' found" % name)
        else:
            self.entries[name] = entry