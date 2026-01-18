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
def generate_import_star(self, env, code):
    env.use_utility_code(UtilityCode.load_cached('CStringEquals', 'StringTools.c'))
    code.putln()
    code.enter_cfunc_scope()
    code.putln('static int %s(PyObject *o, PyObject* py_name, char *name) {' % Naming.import_star_set)
    code.putln('static const char* internal_type_names[] = {')
    for name, entry in sorted(env.entries.items()):
        if entry.is_type:
            code.putln('"%s",' % name)
    code.putln('0')
    code.putln('};')
    code.putln('const char** type_name = internal_type_names;')
    code.putln('while (*type_name) {')
    code.putln('if (__Pyx_StrEq(name, *type_name)) {')
    code.putln('PyErr_Format(PyExc_TypeError, "Cannot overwrite C type %s", name);')
    code.putln('goto bad;')
    code.putln('}')
    code.putln('type_name++;')
    code.putln('}')
    old_error_label = code.new_error_label()
    code.putln('if (0);')
    msvc_count = 0
    for name, entry in sorted(env.entries.items()):
        if entry.is_cglobal and entry.used and (not entry.type.is_const):
            msvc_count += 1
            if msvc_count % 100 == 0:
                code.putln('#ifdef _MSC_VER')
                code.putln('if (0);  /* Workaround for MSVC C1061. */')
                code.putln('#endif')
            code.putln('else if (__Pyx_StrEq(name, "%s")) {' % name)
            if entry.type.is_pyobject:
                if entry.type.is_extension_type or entry.type.is_builtin_type:
                    code.putln('if (!(%s)) %s;' % (entry.type.type_test_code('o'), code.error_goto(entry.pos)))
                code.putln('Py_INCREF(o);')
                code.put_decref(entry.cname, entry.type, nanny=False)
                code.putln('%s = %s;' % (entry.cname, PyrexTypes.typecast(entry.type, py_object_type, 'o')))
            elif entry.type.create_from_py_utility_code(env):
                code.putln(entry.type.from_py_call_code('o', entry.cname, entry.pos, code))
            else:
                code.putln('PyErr_Format(PyExc_TypeError, "Cannot convert Python object %s to %s");' % (name, entry.type))
                code.putln(code.error_goto(entry.pos))
            code.putln('}')
    code.putln('else {')
    code.putln('if (PyObject_SetAttr(%s, py_name, o) < 0) goto bad;' % Naming.module_cname)
    code.putln('}')
    code.putln('return 0;')
    if code.label_used(code.error_label):
        code.put_label(code.error_label)
        code.put_add_traceback(EncodedString(self.full_module_name))
    code.error_label = old_error_label
    code.putln('bad:')
    code.putln('return -1;')
    code.putln('}')
    code.putln('')
    code.putln(UtilityCode.load_as_string('ImportStar', 'ImportExport.c')[1])
    code.exit_cfunc_scope()