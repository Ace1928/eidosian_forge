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
def generate_type_header_code(self, type_entries, code):
    for entry in type_entries:
        if not entry.in_cinclude:
            type = entry.type
            if type.is_typedef:
                pass
            elif type.is_struct_or_union or type.is_cpp_class:
                self.generate_struct_union_predeclaration(entry, code)
            elif type.is_ctuple and entry.used:
                self.generate_struct_union_predeclaration(entry.type.struct_entry, code)
            elif type.is_extension_type:
                self.generate_objstruct_predeclaration(type, code)
    for entry in type_entries:
        if not entry.in_cinclude:
            type = entry.type
            if type.is_typedef:
                self.generate_typedef(entry, code)
            elif type.is_enum or type.is_cpp_enum:
                self.generate_enum_definition(entry, code)
            elif type.is_struct_or_union:
                self.generate_struct_union_definition(entry, code)
            elif type.is_ctuple and entry.used:
                self.generate_struct_union_definition(entry.type.struct_entry, code)
            elif type.is_cpp_class:
                self.generate_cpp_class_definition(entry, code)
            elif type.is_extension_type:
                self.generate_objstruct_definition(type, code)