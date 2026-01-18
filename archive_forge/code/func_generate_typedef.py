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
def generate_typedef(self, entry, code):
    base_type = entry.type.typedef_base_type
    enclosing_scope = entry.scope
    if base_type.is_numeric and (not enclosing_scope.is_cpp_class_scope):
        try:
            writer = code.globalstate['numeric_typedefs']
        except KeyError:
            writer = code
    else:
        writer = code
    writer.mark_pos(entry.pos)
    writer.putln('typedef %s;' % base_type.declaration_code(entry.cname))