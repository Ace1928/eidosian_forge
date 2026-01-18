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
def generate_del_function(self, scope, code):
    tp_slot = TypeSlots.get_slot_by_name('tp_finalize', scope.directives)
    slot_func_cname = scope.mangle_internal('tp_finalize')
    if tp_slot.slot_code(scope) != slot_func_cname:
        return
    entry = scope.lookup_here('__del__')
    if entry is None or not entry.is_special:
        return
    code.putln('')
    if tp_slot.used_ifdef:
        code.putln('#if %s' % tp_slot.used_ifdef)
    code.putln('static void %s(PyObject *o) {' % slot_func_cname)
    code.putln('PyObject *etype, *eval, *etb;')
    code.putln('PyErr_Fetch(&etype, &eval, &etb);')
    code.putln('%s(o);' % entry.func_cname)
    code.putln('PyErr_Restore(etype, eval, etb);')
    code.putln('}')
    if tp_slot.used_ifdef:
        code.putln('#endif')