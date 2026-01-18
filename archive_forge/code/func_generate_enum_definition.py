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
def generate_enum_definition(self, entry, code):
    code.mark_pos(entry.pos)
    type = entry.type
    name = entry.cname or entry.name or ''
    kind = 'enum class' if entry.type.is_cpp_enum else 'enum'
    header, footer = self.sue_header_footer(type, kind, name)
    code.putln(header)
    enum_values = entry.enum_values
    if not enum_values:
        error(entry.pos, "Empty enum definition not allowed outside a 'cdef extern from' block")
    else:
        last_entry = enum_values[-1]
        for value_entry in enum_values:
            if value_entry.value_node is not None:
                value_entry.value_node.generate_evaluation_code(code)
        for value_entry in enum_values:
            if value_entry.value_node is None:
                value_code = value_entry.cname.split('::')[-1]
            else:
                value_code = '%s = %s' % (value_entry.cname.split('::')[-1], value_entry.value_node.result())
            if value_entry is not last_entry:
                value_code += ','
            code.putln(value_code)
    code.putln(footer)
    if entry.type.is_enum:
        if entry.type.typedef_flag:
            code.putln('typedef enum %s %s;' % (name, name))