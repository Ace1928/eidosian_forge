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
def generate_struct_union_definition(self, entry, code):
    code.mark_pos(entry.pos)
    type = entry.type
    scope = type.scope
    if scope:
        kind = type.kind
        packed = type.is_struct and type.packed
        if packed:
            kind = '%s %s' % (type.kind, '__Pyx_PACKED')
            code.globalstate.use_utility_code(packed_struct_utility_code)
        header, footer = self.sue_header_footer(type, kind, type.cname)
        if packed:
            code.putln('#if defined(__SUNPRO_C)')
            code.putln('  #pragma pack(1)')
            code.putln('#elif !defined(__GNUC__)')
            code.putln('  #pragma pack(push, 1)')
            code.putln('#endif')
        code.putln(header)
        var_entries = scope.var_entries
        for attr in var_entries:
            code.putln('%s;' % attr.type.declaration_code(attr.cname))
        code.putln(footer)
        if packed:
            code.putln('#if defined(__SUNPRO_C)')
            code.putln('  #pragma pack()')
            code.putln('#elif !defined(__GNUC__)')
            code.putln('  #pragma pack(pop)')
            code.putln('#endif')