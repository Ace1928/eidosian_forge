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
def find_imported_module(self, path, pos):
    scope = self
    for name in path:
        entry = scope.find(name, pos)
        if not entry:
            return None
        if entry.as_module:
            scope = entry.as_module
        else:
            error(pos, "'%s' is not a cimported module" % '.'.join(path))
            return None
    return scope