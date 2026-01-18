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
def generate_cfunction_declaration(entry, env, code, definition):
    from_cy_utility = entry.used and entry.utility_code_definition
    if entry.used and entry.inline_func_in_pxd or (not entry.in_cinclude and (definition or entry.defined_in_pxd or entry.visibility == 'extern' or from_cy_utility)):
        if entry.visibility == 'extern':
            storage_class = Naming.extern_c_macro
            dll_linkage = 'DL_IMPORT'
        elif entry.visibility == 'public':
            storage_class = Naming.extern_c_macro
            dll_linkage = None
        elif entry.visibility == 'private':
            storage_class = 'static'
            dll_linkage = None
        else:
            storage_class = 'static'
            dll_linkage = None
        type = entry.type
        if entry.defined_in_pxd and (not definition):
            storage_class = 'static'
            dll_linkage = None
            type = CPtrType(type)
        header = type.declaration_code(entry.cname, dll_linkage=dll_linkage)
        modifiers = code.build_function_modifiers(entry.func_modifiers)
        code.putln('%s %s%s; /*proto*/' % (storage_class, modifiers, header))