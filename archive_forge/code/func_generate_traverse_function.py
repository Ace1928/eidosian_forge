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
def generate_traverse_function(self, scope, code, cclass_entry):
    tp_slot = TypeSlots.GCDependentSlot('tp_traverse')
    slot_func = scope.mangle_internal('tp_traverse')
    base_type = scope.parent_type.base_type
    if tp_slot.slot_code(scope) != slot_func:
        return
    code.putln('')
    code.putln('static int %s(PyObject *o, visitproc v, void *a) {' % slot_func)
    have_entries, (py_attrs, py_buffers, memoryview_slices) = scope.get_refcounted_entries(include_gc_simple=False)
    if base_type or py_attrs:
        code.putln('int e;')
    if py_attrs or py_buffers:
        self.generate_self_cast(scope, code)
    if base_type:
        static_call = TypeSlots.get_base_slot_function(scope, tp_slot)
        if static_call:
            code.putln('e = %s(o, v, a); if (e) return e;' % static_call)
        elif base_type.is_builtin_type:
            base_cname = base_type.typeptr_cname
            code.putln('if (!%s->tp_traverse); else { e = %s->tp_traverse(o,v,a); if (e) return e; }' % (base_cname, base_cname))
        else:
            base_cname = base_type.typeptr_cname
            code.putln('e = ((likely(%s)) ? ((%s->tp_traverse) ? %s->tp_traverse(o, v, a) : 0) : __Pyx_call_next_tp_traverse(o, v, a, %s)); if (e) return e;' % (base_cname, base_cname, base_cname, slot_func))
            code.globalstate.use_utility_code(UtilityCode.load_cached('CallNextTpTraverse', 'ExtensionTypes.c'))
    for entry in py_attrs:
        var_code = 'p->%s' % entry.cname
        var_as_pyobject = PyrexTypes.typecast(py_object_type, entry.type, var_code)
        code.putln('if (%s) {' % var_code)
        code.putln('e = (*v)(%s, a); if (e) return e;' % var_as_pyobject)
        code.putln('}')
    for entry in py_buffers:
        cname = entry.cname + '.obj'
        code.putln('if (p->%s) {' % cname)
        code.putln('e = (*v)(p->%s, a); if (e) return e;' % cname)
        code.putln('}')
    code.putln('return 0;')
    code.putln('}')