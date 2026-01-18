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
def declare_inherited_cpp_attributes(self, base_class):
    base_scope = base_class.scope
    template_type = base_class
    while getattr(template_type, 'template_type', None):
        template_type = template_type.template_type
    if getattr(template_type, 'templates', None):
        base_templates = [T.name for T in template_type.templates]
    else:
        base_templates = ()
    for base_entry in base_scope.inherited_var_entries + base_scope.var_entries:
        if base_entry.name in ('<init>', '<del>'):
            continue
        if base_entry.name in self.entries:
            base_entry.name
        entry = self.declare(base_entry.name, base_entry.cname, base_entry.type, None, 'extern')
        entry.is_variable = 1
        entry.is_inherited = 1
        self.inherited_var_entries.append(entry)
    for base_entry in base_scope.cfunc_entries:
        entry = self.declare_cfunction(base_entry.name, base_entry.type, base_entry.pos, base_entry.cname, base_entry.visibility, api=0, modifiers=base_entry.func_modifiers, utility_code=base_entry.utility_code)
        entry.is_inherited = 1
    for base_entry in base_scope.type_entries:
        if base_entry.name not in base_templates:
            entry = self.declare_type(base_entry.name, base_entry.type, base_entry.pos, base_entry.cname, base_entry.visibility, defining=False)
            entry.is_inherited = 1