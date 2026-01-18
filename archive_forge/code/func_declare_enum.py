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
def declare_enum(self, name, pos, cname, scoped, typedef_flag, visibility='private', api=0, create_wrapper=0, doc=None):
    if name:
        if not cname:
            if self.in_cinclude or visibility == 'public' or visibility == 'extern' or api:
                cname = name
            else:
                cname = self.mangle(Naming.type_prefix, name)
        if self.is_cpp_class_scope:
            namespace = self.outer_scope.lookup(self.name).type
        else:
            namespace = None
        if scoped:
            type = PyrexTypes.CppScopedEnumType(name, cname, namespace, doc=doc)
        else:
            type = PyrexTypes.CEnumType(name, cname, typedef_flag, namespace, doc=doc)
    else:
        type = PyrexTypes.c_anon_enum_type
    entry = self.declare_type(name, type, pos, cname=cname, visibility=visibility, api=api)
    if scoped:
        entry.utility_code = Code.UtilityCode.load_cached('EnumClassDecl', 'CppSupport.cpp')
        self.use_entry_utility_code(entry)
    entry.create_wrapper = create_wrapper
    entry.enum_values = []
    self.sue_entries.append(entry)
    return entry