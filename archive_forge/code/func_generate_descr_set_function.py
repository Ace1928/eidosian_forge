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
def generate_descr_set_function(self, scope, code):
    base_type = scope.parent_type.base_type
    user_set_entry = scope.lookup_here('__set__')
    user_del_entry = scope.lookup_here('__delete__')
    code.putln('')
    code.putln('static int %s(PyObject *o, PyObject *i, PyObject *v) {' % scope.mangle_internal('tp_descr_set'))
    code.putln('if (v) {')
    if user_set_entry:
        code.putln('return %s(o, i, v);' % user_set_entry.func_cname)
    else:
        self.generate_guarded_basetype_call(base_type, None, 'tp_descr_set', 'o, i, v', code)
        code.putln('PyErr_SetString(PyExc_NotImplementedError, "__set__");')
        code.putln('return -1;')
    code.putln('}')
    code.putln('else {')
    if user_del_entry:
        code.putln('return %s(o, i);' % user_del_entry.func_cname)
    else:
        self.generate_guarded_basetype_call(base_type, None, 'tp_descr_set', 'o, i, v', code)
        code.putln('PyErr_SetString(PyExc_NotImplementedError, "__delete__");')
        code.putln('return -1;')
    code.putln('}')
    code.putln('}')