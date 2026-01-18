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
def generate_binop_function(self, scope, slot, code, pos):
    func_name = scope.mangle_internal(slot.slot_name)
    if scope.directives['c_api_binop_methods']:
        code.putln('#define %s %s' % (func_name, slot.left_slot.slot_code(scope)))
        return
    code.putln()
    preprocessor_guard = slot.preprocessor_guard_code()
    if preprocessor_guard:
        code.putln(preprocessor_guard)
    if slot.left_slot.signature in (TypeSlots.binaryfunc, TypeSlots.ibinaryfunc):
        slot_type = 'binaryfunc'
        extra_arg = extra_arg_decl = ''
    elif slot.left_slot.signature in (TypeSlots.powternaryfunc, TypeSlots.ipowternaryfunc):
        slot_type = 'ternaryfunc'
        extra_arg = ', extra_arg'
        extra_arg_decl = ', PyObject* extra_arg'
    else:
        error(pos, 'Unexpected type slot signature: %s' % slot)
        return

    def get_slot_method_cname(method_name):
        entry = scope.lookup(method_name)
        return entry.func_cname if entry and entry.is_special else None

    def call_slot_method(method_name, reverse):
        func_cname = get_slot_method_cname(method_name)
        if func_cname:
            return '%s(%s%s)' % (func_cname, 'right, left' if reverse else 'left, right', extra_arg)
        else:
            return '%s_maybe_call_slot(__Pyx_PyType_GetSlot(%s, tp_base, PyTypeObject*), left, right %s)' % (func_name, scope.parent_type.typeptr_cname, extra_arg)
    if get_slot_method_cname(slot.left_slot.method_name) and (not get_slot_method_cname(slot.right_slot.method_name)):
        warning(pos, 'Extension type implements %s() but not %s(). The behaviour has changed from previous Cython versions to match Python semantics. You can implement both special methods in a backwards compatible way.' % (slot.left_slot.method_name, slot.right_slot.method_name))
    overloads_left = int(bool(get_slot_method_cname(slot.left_slot.method_name)))
    overloads_right = int(bool(get_slot_method_cname(slot.right_slot.method_name)))
    code.putln(TempitaUtilityCode.load_as_string('BinopSlot', 'ExtensionTypes.c', context={'func_name': func_name, 'slot_name': slot.slot_name, 'overloads_left': overloads_left, 'overloads_right': overloads_right, 'call_left': call_slot_method(slot.left_slot.method_name, reverse=False), 'call_right': call_slot_method(slot.right_slot.method_name, reverse=True), 'type_cname': scope.parent_type.typeptr_cname, 'slot_type': slot_type, 'extra_arg': extra_arg, 'extra_arg_decl': extra_arg_decl})[1])
    if preprocessor_guard:
        code.putln('#endif')