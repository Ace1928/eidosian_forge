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
def generate_dict_getter_function(self, scope, code):
    dict_attr = scope.lookup_here('__dict__')
    if not dict_attr or not dict_attr.is_variable:
        return
    func_name = scope.mangle_internal('__dict__getter')
    dict_name = dict_attr.cname
    code.putln('')
    code.putln('static PyObject *%s(PyObject *o, CYTHON_UNUSED void *x) {' % func_name)
    self.generate_self_cast(scope, code)
    code.putln('if (unlikely(!p->%s)){' % dict_name)
    code.putln('p->%s = PyDict_New();' % dict_name)
    code.putln('}')
    code.putln('Py_XINCREF(p->%s);' % dict_name)
    code.putln('return p->%s;' % dict_name)
    code.putln('}')