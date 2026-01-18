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
def generate_module_state_start(self, env, code):
    code.putln('typedef struct {')
    code.putln('PyObject *%s;' % env.module_dict_cname)
    code.putln('PyObject *%s;' % Naming.builtins_cname)
    code.putln('PyObject *%s;' % Naming.cython_runtime_cname)
    code.putln('PyObject *%s;' % Naming.empty_tuple)
    code.putln('PyObject *%s;' % Naming.empty_bytes)
    code.putln('PyObject *%s;' % Naming.empty_unicode)
    if Options.pre_import is not None:
        code.putln('PyObject *%s;' % Naming.preimport_cname)
    for type_cname, used_name in Naming.used_types_and_macros:
        code.putln('#ifdef %s' % used_name)
        code.putln('PyTypeObject *%s;' % type_cname)
        code.putln('#endif')