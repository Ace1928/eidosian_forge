from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def generate_typeobj_definitions(self, env, code):
    full_module_name = env.qualified_name
    for entry in env.c_class_entries:
        if entry.visibility != 'extern':
            type = entry.type
            scope = type.scope
            if scope:
                self.generate_exttype_vtable(scope, code)
                self.generate_new_function(scope, code, entry)
                self.generate_del_function(scope, code)
                self.generate_dealloc_function(scope, code)
                if scope.needs_gc():
                    self.generate_traverse_function(scope, code, entry)
                    if scope.needs_tp_clear():
                        self.generate_clear_function(scope, code, entry)
                if scope.defines_any_special(['__getitem__']):
                    self.generate_getitem_int_function(scope, code)
                if scope.defines_any_special(['__setitem__', '__delitem__']):
                    self.generate_ass_subscript_function(scope, code)
                if scope.defines_any_special(['__getslice__', '__setslice__', '__delslice__']):
                    warning(self.pos, '__getslice__, __setslice__, and __delslice__ are not supported by Python 3, use __getitem__, __setitem__, and __delitem__ instead', 1)
                    code.putln('#if PY_MAJOR_VERSION >= 3')
                    code.putln('#error __getslice__, __setslice__, and __delslice__ not supported in Python 3.')
                    code.putln('#endif')
                if scope.defines_any_special(['__setslice__', '__delslice__']):
                    self.generate_ass_slice_function(scope, code)
                if scope.defines_any_special(['__getattr__', '__getattribute__']):
                    self.generate_getattro_function(scope, code)
                if scope.defines_any_special(['__setattr__', '__delattr__']):
                    self.generate_setattro_function(scope, code)
                if scope.defines_any_special(['__get__']):
                    self.generate_descr_get_function(scope, code)
                if scope.defines_any_special(['__set__', '__delete__']):
                    self.generate_descr_set_function(scope, code)
                if not scope.is_closure_class_scope and scope.defines_any(['__dict__']):
                    self.generate_dict_getter_function(scope, code)
                if scope.defines_any_special(TypeSlots.richcmp_special_methods):
                    self.generate_richcmp_function(scope, code)
                elif 'total_ordering' in scope.directives:
                    warning(scope.parent_type.pos, 'total_ordering directive used, but no comparison and equality methods defined')
                for slot in TypeSlots.get_slot_table(code.globalstate.directives).PyNumberMethods:
                    if slot.is_binop and scope.defines_any_special(slot.user_methods):
                        self.generate_binop_function(scope, slot, code, entry.pos)
                self.generate_property_accessors(scope, code)
                self.generate_method_table(scope, code)
                self.generate_getset_table(scope, code)
                code.putln('#if CYTHON_USE_TYPE_SPECS')
                self.generate_typeobj_spec(entry, code)
                code.putln('#else')
                self.generate_typeobj_definition(full_module_name, entry, code)
                code.putln('#endif')