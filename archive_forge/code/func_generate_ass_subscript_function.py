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
def generate_ass_subscript_function(self, scope, code):
    base_type = scope.parent_type.base_type
    set_entry = scope.lookup_here('__setitem__')
    del_entry = scope.lookup_here('__delitem__')
    code.putln('')
    code.putln('static int %s(PyObject *o, PyObject *i, PyObject *v) {' % scope.mangle_internal('mp_ass_subscript'))
    code.putln('if (v) {')
    if set_entry:
        code.putln('return %s(o, i, v);' % set_entry.func_cname)
    else:
        code.putln('__Pyx_TypeName o_type_name;')
        self.generate_guarded_basetype_call(base_type, 'tp_as_mapping', 'mp_ass_subscript', 'o, i, v', code)
        code.putln('o_type_name = __Pyx_PyType_GetName(Py_TYPE(o));')
        code.putln('PyErr_Format(PyExc_NotImplementedError,')
        code.putln('  "Subscript assignment not supported by " __Pyx_FMT_TYPENAME, o_type_name);')
        code.putln('__Pyx_DECREF_TypeName(o_type_name);')
        code.putln('return -1;')
    code.putln('}')
    code.putln('else {')
    if del_entry:
        code.putln('return %s(o, i);' % del_entry.func_cname)
    else:
        code.putln('__Pyx_TypeName o_type_name;')
        self.generate_guarded_basetype_call(base_type, 'tp_as_mapping', 'mp_ass_subscript', 'o, i, v', code)
        code.putln('o_type_name = __Pyx_PyType_GetName(Py_TYPE(o));')
        code.putln('PyErr_Format(PyExc_NotImplementedError,')
        code.putln('  "Subscript deletion not supported by " __Pyx_FMT_TYPENAME, o_type_name);')
        code.putln('__Pyx_DECREF_TypeName(o_type_name);')
        code.putln('return -1;')
    code.putln('}')
    code.putln('}')