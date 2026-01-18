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
def mangle_closure_cnames(self, outer_scope_cname):
    for scope in self.iter_local_scopes():
        for entry in scope.entries.values():
            if entry.from_closure:
                cname = entry.outer_entry.cname
                if self.is_passthrough:
                    entry.cname = cname
                else:
                    if cname.startswith(Naming.cur_scope_cname):
                        cname = cname[len(Naming.cur_scope_cname) + 2:]
                    entry.cname = '%s->%s' % (outer_scope_cname, cname)
            elif entry.in_closure:
                entry.original_cname = entry.cname
                entry.cname = '%s->%s' % (Naming.cur_scope_cname, entry.cname)
                if entry.type.is_cpp_class and entry.scope.directives['cpp_locals']:
                    entry.make_cpp_optional()