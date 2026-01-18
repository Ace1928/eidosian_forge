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
def generate_typeobj_spec(self, entry, code):
    ext_type = entry.type
    scope = ext_type.scope
    members_slot = TypeSlots.get_slot_by_name('tp_members', code.globalstate.directives)
    members_slot.generate_substructure_spec(scope, code)
    buffer_slot = TypeSlots.get_slot_by_name('tp_as_buffer', code.globalstate.directives)
    if not buffer_slot.is_empty(scope):
        code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
        buffer_slot.generate_substructure(scope, code)
        code.putln('#endif')
    code.putln('static PyType_Slot %s_slots[] = {' % ext_type.typeobj_cname)
    for slot in TypeSlots.get_slot_table(code.globalstate.directives):
        slot.generate_spec(scope, code)
    code.putln('{0, 0},')
    code.putln('};')
    if ext_type.typedef_flag:
        objstruct = ext_type.objstruct_cname
    else:
        objstruct = 'struct %s' % ext_type.objstruct_cname
    classname = scope.class_name.as_c_string_literal()
    code.putln('static PyType_Spec %s_spec = {' % ext_type.typeobj_cname)
    code.putln('"%s.%s",' % (self.full_module_name, classname.replace('"', '')))
    code.putln('sizeof(%s),' % objstruct)
    code.putln('0,')
    code.putln('%s,' % TypeSlots.get_slot_by_name('tp_flags', scope.directives).slot_code(scope))
    code.putln('%s_slots,' % ext_type.typeobj_cname)
    code.putln('};')