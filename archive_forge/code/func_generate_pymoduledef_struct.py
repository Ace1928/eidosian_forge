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
def generate_pymoduledef_struct(self, env, code):
    if env.doc:
        doc = '%s' % code.get_string_const(env.doc)
    else:
        doc = '0'
    if Options.generate_cleanup_code:
        cleanup_func = '(freefunc)%s' % Naming.cleanup_cname
    else:
        cleanup_func = 'NULL'
    code.putln('')
    code.putln('#if PY_MAJOR_VERSION >= 3')
    code.putln('#if CYTHON_PEP489_MULTI_PHASE_INIT')
    exec_func_cname = self.module_init_func_cname()
    code.putln('static PyObject* %s(PyObject *spec, PyModuleDef *def); /*proto*/' % Naming.pymodule_create_func_cname)
    code.putln('static int %s(PyObject* module); /*proto*/' % exec_func_cname)
    code.putln('static PyModuleDef_Slot %s[] = {' % Naming.pymoduledef_slots_cname)
    code.putln('{Py_mod_create, (void*)%s},' % Naming.pymodule_create_func_cname)
    code.putln('{Py_mod_exec, (void*)%s},' % exec_func_cname)
    code.putln('{0, NULL}')
    code.putln('};')
    if not env.module_name.isascii():
        code.putln('#else /* CYTHON_PEP489_MULTI_PHASE_INIT */')
        code.putln('#error "Unicode module names are only supported with multi-phase init as per PEP489"')
    code.putln('#endif')
    code.putln('')
    code.putln('#ifdef __cplusplus')
    code.putln('namespace {')
    code.putln('struct PyModuleDef %s =' % Naming.pymoduledef_cname)
    code.putln('#else')
    code.putln('static struct PyModuleDef %s =' % Naming.pymoduledef_cname)
    code.putln('#endif')
    code.putln('{')
    code.putln('  PyModuleDef_HEAD_INIT,')
    code.putln('  %s,' % env.module_name.as_c_string_literal())
    code.putln('  %s, /* m_doc */' % doc)
    code.putln('#if CYTHON_PEP489_MULTI_PHASE_INIT')
    code.putln('  0, /* m_size */')
    code.putln('#elif CYTHON_USE_MODULE_STATE')
    code.putln('  sizeof(%s), /* m_size */' % Naming.modulestate_cname)
    code.putln('#else')
    code.putln('  -1, /* m_size */')
    code.putln('#endif')
    code.putln('  %s /* m_methods */,' % env.method_table_cname)
    code.putln('#if CYTHON_PEP489_MULTI_PHASE_INIT')
    code.putln('  %s, /* m_slots */' % Naming.pymoduledef_slots_cname)
    code.putln('#else')
    code.putln('  NULL, /* m_reload */')
    code.putln('#endif')
    code.putln('#if CYTHON_USE_MODULE_STATE')
    code.putln('  %s_traverse, /* m_traverse */' % Naming.module_cname)
    code.putln('  %s_clear, /* m_clear */' % Naming.module_cname)
    code.putln('  %s /* m_free */' % cleanup_func)
    code.putln('#else')
    code.putln('  NULL, /* m_traverse */')
    code.putln('  NULL, /* m_clear */')
    code.putln('  %s /* m_free */' % cleanup_func)
    code.putln('#endif')
    code.putln('};')
    code.putln('#ifdef __cplusplus')
    code.putln('} /* anonymous namespace */')
    code.putln('#endif')
    code.putln('#endif')