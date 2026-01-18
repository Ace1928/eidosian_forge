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
def sort_type_hierarchy(self, module_list, env):
    vtab_dict, vtab_dict_order = ({}, [])
    vtabslot_dict, vtabslot_dict_order = ({}, [])
    for module in module_list:
        for entry in module.c_class_entries:
            if entry.used and (not entry.in_cinclude):
                type = entry.type
                key = type.vtabstruct_cname
                if not key:
                    continue
                if key in vtab_dict:
                    from .UtilityCode import NonManglingModuleScope
                    assert isinstance(entry.scope, NonManglingModuleScope), str(entry.scope)
                    assert isinstance(vtab_dict[key].scope, NonManglingModuleScope), str(vtab_dict[key].scope)
                else:
                    vtab_dict[key] = entry
                    vtab_dict_order.append(key)
        all_defined_here = module is env
        for entry in module.type_entries:
            if entry.used and (all_defined_here or entry.defined_in_pxd):
                type = entry.type
                if type.is_extension_type and (not entry.in_cinclude):
                    type = entry.type
                    key = type.objstruct_cname
                    assert key not in vtabslot_dict, key
                    vtabslot_dict[key] = entry
                    vtabslot_dict_order.append(key)

    def vtabstruct_cname(entry_type):
        return entry_type.vtabstruct_cname
    vtab_list = self.sort_types_by_inheritance(vtab_dict, vtab_dict_order, vtabstruct_cname)

    def objstruct_cname(entry_type):
        return entry_type.objstruct_cname
    vtabslot_list = self.sort_types_by_inheritance(vtabslot_dict, vtabslot_dict_order, objstruct_cname)
    return (vtab_list, vtabslot_list)