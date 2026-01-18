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
def generate_module_creation_code(self, env, code):
    if env.doc:
        doc = '%s' % code.get_string_const(env.doc)
    else:
        doc = '0'
    code.putln('#if CYTHON_PEP489_MULTI_PHASE_INIT')
    code.putln('%s = %s;' % (env.module_cname, Naming.pymodinit_module_arg))
    code.put_incref(env.module_cname, py_object_type, nanny=False)
    code.putln('#else')
    code.putln('#if PY_MAJOR_VERSION < 3')
    code.putln('%s = Py_InitModule4(%s, %s, %s, 0, PYTHON_API_VERSION); Py_XINCREF(%s);' % (env.module_cname, env.module_name.as_c_string_literal(), env.method_table_cname, doc, env.module_cname))
    code.putln(code.error_goto_if_null(env.module_cname, self.pos))
    code.putln('#elif CYTHON_USE_MODULE_STATE')
    module_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=False)
    code.putln('%s = PyModule_Create(&%s); %s' % (module_temp, Naming.pymoduledef_cname, code.error_goto_if_null(module_temp, self.pos)))
    code.putln('{')
    code.putln('int add_module_result = PyState_AddModule(%s, &%s);' % (module_temp, Naming.pymoduledef_cname))
    code.putln('%s = 0; /* transfer ownership from %s to %s pseudovariable */' % (module_temp, module_temp, env.module_name.as_c_string_literal()))
    code.putln(code.error_goto_if_neg('add_module_result', self.pos))
    code.putln('pystate_addmodule_run = 1;')
    code.putln('}')
    code.funcstate.release_temp(module_temp)
    code.putln('#else')
    code.putln('%s = PyModule_Create(&%s);' % (env.module_cname, Naming.pymoduledef_cname))
    code.putln(code.error_goto_if_null(env.module_cname, self.pos))
    code.putln('#endif')
    code.putln('#endif')
    code.putln('CYTHON_UNUSED_VAR(%s);' % module_temp)
    code.putln('%s = PyModule_GetDict(%s); %s' % (env.module_dict_cname, env.module_cname, code.error_goto_if_null(env.module_dict_cname, self.pos)))
    code.put_incref(env.module_dict_cname, py_object_type, nanny=False)
    code.putln('%s = __Pyx_PyImport_AddModuleRef(__Pyx_BUILTIN_MODULE_NAME); %s' % (Naming.builtins_cname, code.error_goto_if_null(Naming.builtins_cname, self.pos)))
    code.putln('%s = __Pyx_PyImport_AddModuleRef((const char *) "cython_runtime"); %s' % (Naming.cython_runtime_cname, code.error_goto_if_null(Naming.cython_runtime_cname, self.pos)))
    code.putln('if (PyObject_SetAttrString(%s, "__builtins__", %s) < 0) %s' % (env.module_cname, Naming.builtins_cname, code.error_goto(self.pos)))
    if Options.pre_import is not None:
        code.putln('%s = __Pyx_PyImport_AddModuleRef("%s"); %s' % (Naming.preimport_cname, Options.pre_import, code.error_goto_if_null(Naming.preimport_cname, self.pos)))