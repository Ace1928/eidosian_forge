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
def generate_module_state_clear(self, env, code):
    code.putln('#if CYTHON_USE_MODULE_STATE')
    code.putln('static int %s_clear(PyObject *m) {' % Naming.module_cname)
    code.putln('%s *clear_module_state = %s(m);' % (Naming.modulestate_cname, Naming.modulestate_cname))
    code.putln('if (!clear_module_state) return 0;')
    code.putln('Py_CLEAR(clear_module_state->%s);' % env.module_dict_cname)
    code.putln('Py_CLEAR(clear_module_state->%s);' % Naming.builtins_cname)
    code.putln('Py_CLEAR(clear_module_state->%s);' % Naming.cython_runtime_cname)
    code.putln('Py_CLEAR(clear_module_state->%s);' % Naming.empty_tuple)
    code.putln('Py_CLEAR(clear_module_state->%s);' % Naming.empty_bytes)
    code.putln('Py_CLEAR(clear_module_state->%s);' % Naming.empty_unicode)
    code.putln('#ifdef __Pyx_CyFunction_USED')
    code.putln('Py_CLEAR(clear_module_state->%s);' % Naming.cyfunction_type_cname)
    code.putln('#endif')
    code.putln('#ifdef __Pyx_FusedFunction_USED')
    code.putln('Py_CLEAR(clear_module_state->%s);' % Naming.fusedfunction_type_cname)
    code.putln('#endif')