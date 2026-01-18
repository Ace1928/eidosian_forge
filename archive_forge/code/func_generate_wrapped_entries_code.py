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
def generate_wrapped_entries_code(self, env, code):
    for name, entry in sorted(env.entries.items()):
        if entry.create_wrapper and (not entry.is_type) and (entry.scope is env):
            if not entry.type.create_to_py_utility_code(env):
                error(entry.pos, "Cannot convert '%s' to Python object" % entry.type)
            code.putln('{')
            code.putln('PyObject* wrapped = %s(%s);' % (entry.type.to_py_function, entry.cname))
            code.putln(code.error_goto_if_null('wrapped', entry.pos))
            code.putln('if (PyObject_SetAttrString(%s, "%s", wrapped) < 0) %s;' % (env.module_cname, name, code.error_goto(entry.pos)))
            code.putln('}')