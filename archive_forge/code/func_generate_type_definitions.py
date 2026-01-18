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
def generate_type_definitions(self, env, modules, vtab_list, vtabslot_list, code):
    for entry in vtabslot_list:
        self.generate_objstruct_predeclaration(entry.type, code)
    vtabslot_entries = set(vtabslot_list)
    ctuple_names = set()
    for module in modules:
        definition = module is env
        type_entries = []
        for entry in module.type_entries:
            if entry.type.is_ctuple and entry.used:
                if entry.name not in ctuple_names:
                    ctuple_names.add(entry.name)
                    type_entries.append(entry)
            elif definition or entry.defined_in_pxd:
                type_entries.append(entry)
        type_entries = [t for t in type_entries if t not in vtabslot_entries]
        self.generate_type_header_code(type_entries, code)
    for entry in vtabslot_list:
        self.generate_objstruct_definition(entry.type, code)
        self.generate_typeobj_predeclaration(entry, code)
    for entry in vtab_list:
        self.generate_typeobj_predeclaration(entry, code)
        self.generate_exttype_vtable_struct(entry, code)
        self.generate_exttype_vtabptr_declaration(entry, code)
        self.generate_exttype_final_methods_declaration(entry, code)