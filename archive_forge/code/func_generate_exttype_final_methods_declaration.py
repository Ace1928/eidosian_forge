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
def generate_exttype_final_methods_declaration(self, entry, code):
    if not entry.used:
        return
    code.mark_pos(entry.pos)
    type = entry.type
    for method_entry in entry.type.scope.cfunc_entries:
        if not method_entry.is_inherited and method_entry.final_func_cname:
            declaration = method_entry.type.declaration_code(method_entry.final_func_cname)
            modifiers = code.build_function_modifiers(method_entry.func_modifiers)
            code.putln('static %s%s;' % (modifiers, declaration))