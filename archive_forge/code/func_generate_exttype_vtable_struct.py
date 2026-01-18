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
def generate_exttype_vtable_struct(self, entry, code):
    if not entry.used:
        return
    code.mark_pos(entry.pos)
    type = entry.type
    scope = type.scope
    self.specialize_fused_types(scope)
    if type.vtabstruct_cname:
        code.putln('')
        code.putln('struct %s {' % type.vtabstruct_cname)
        if type.base_type and type.base_type.vtabstruct_cname:
            code.putln('struct %s %s;' % (type.base_type.vtabstruct_cname, Naming.obj_base_cname))
        for method_entry in scope.cfunc_entries:
            if not method_entry.is_inherited:
                code.putln('%s;' % method_entry.type.declaration_code('(*%s)' % method_entry.cname))
        code.putln('};')