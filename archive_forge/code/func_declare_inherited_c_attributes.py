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
def declare_inherited_c_attributes(self, base_scope):

    def adapt(cname):
        return '%s.%s' % (Naming.obj_base_cname, base_entry.cname)
    entries = base_scope.inherited_var_entries + base_scope.var_entries
    for base_entry in entries:
        entry = self.declare(base_entry.name, adapt(base_entry.cname), base_entry.type, None, 'private')
        entry.is_variable = 1
        entry.is_inherited = True
        entry.annotation = base_entry.annotation
        self.inherited_var_entries.append(entry)
    for base_entry in base_scope.cfunc_entries[:]:
        if base_entry.type.is_fused:
            base_entry.type.get_all_specialized_function_types()
    for base_entry in base_scope.cfunc_entries:
        cname = base_entry.cname
        var_entry = base_entry.as_variable
        is_builtin = var_entry and var_entry.is_builtin
        if not is_builtin:
            cname = adapt(cname)
        entry = self.add_cfunction(base_entry.name, base_entry.type, base_entry.pos, cname, base_entry.visibility, base_entry.func_modifiers, inherited=True)
        entry.is_inherited = 1
        if base_entry.is_final_cmethod:
            entry.is_final_cmethod = True
            entry.is_inline_cmethod = base_entry.is_inline_cmethod
            if self.parent_scope == base_scope.parent_scope or entry.is_inline_cmethod:
                entry.final_func_cname = base_entry.final_func_cname
        if is_builtin:
            entry.is_builtin_cmethod = True
            entry.as_variable = var_entry
        if base_entry.utility_code:
            entry.utility_code = base_entry.utility_code