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
def lookup_assignment_expression_target(self, name):
    entry = self.lookup_here(name)
    if not entry:
        entry = self.parent_scope.lookup_assignment_expression_target(name)
        if entry:
            return self._create_inner_entry_for_closure(name, entry)
    return entry