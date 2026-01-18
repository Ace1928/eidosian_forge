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
def declare_builtin_type(self, name, cname, utility_code=None, objstruct_cname=None, type_class=PyrexTypes.BuiltinObjectType):
    name = EncodedString(name)
    type = type_class(name, cname, objstruct_cname)
    scope = CClassScope(name, outer_scope=None, visibility='extern', parent_type=type)
    scope.directives = {}
    if name == 'bool':
        type.is_final_type = True
    type.set_scope(scope)
    self.type_names[name] = 1
    entry = self.declare_type(name, type, None, visibility='extern')
    entry.utility_code = utility_code
    var_entry = Entry(name=entry.name, type=self.lookup('type').type, pos=entry.pos, cname=entry.type.typeptr_cname)
    var_entry.qualified_name = self.qualify_name(name)
    var_entry.is_variable = 1
    var_entry.is_cglobal = 1
    var_entry.is_readonly = 1
    var_entry.is_builtin = 1
    var_entry.utility_code = utility_code
    var_entry.scope = self
    if Options.cache_builtins:
        var_entry.is_const = True
    entry.as_variable = var_entry
    return type