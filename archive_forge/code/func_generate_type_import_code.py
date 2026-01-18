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
def generate_type_import_code(self, env, type, pos, code, import_generator):
    if type in env.types_imported:
        return
    if type.name not in Code.ctypedef_builtins_map:
        code.globalstate.use_utility_code(UtilityCode.load_cached('TypeImport', 'ImportExport.c'))
    self.generate_type_import_call(type, code, import_generator, error_pos=pos)
    if type.vtabptr_cname:
        code.globalstate.use_utility_code(UtilityCode.load_cached('GetVTable', 'ImportExport.c'))
        code.putln('%s = (struct %s*)__Pyx_GetVtable(%s); %s' % (type.vtabptr_cname, type.vtabstruct_cname, type.typeptr_cname, code.error_goto_if_null(type.vtabptr_cname, pos)))
    env.types_imported.add(type)