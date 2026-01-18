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
def allocate_vtable_names(self, entry):
    type = entry.type
    if type.base_type and type.base_type.vtabslot_cname:
        type.vtabslot_cname = '%s.%s' % (Naming.obj_base_cname, type.base_type.vtabslot_cname)
    elif type.scope and type.scope.cfunc_entries:
        entry_count = len(type.scope.cfunc_entries)
        base_type = type.base_type
        while base_type:
            if not base_type.scope or entry_count > len(base_type.scope.cfunc_entries):
                break
            if base_type.is_builtin_type:
                return
            base_type = base_type.base_type
        type.vtabslot_cname = Naming.vtabslot_cname
    if type.vtabslot_cname:
        type.vtabstruct_cname = self.mangle(Naming.vtabstruct_prefix, entry.name)
        type.vtabptr_cname = self.mangle(Naming.vtabptr_prefix, entry.name)