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
def declare_global(self, name, pos):
    if self.lookup_here(name):
        warning(pos, "'%s' redeclared  ", 0)
    else:
        entry = self.global_scope().lookup_target(name)
        self.entries[name] = entry