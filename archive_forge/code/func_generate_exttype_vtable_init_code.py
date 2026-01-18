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
def generate_exttype_vtable_init_code(self, entry, code):
    type = entry.type
    if type.vtable_cname:
        code.putln('%s = &%s;' % (type.vtabptr_cname, type.vtable_cname))
        if type.base_type and type.base_type.vtabptr_cname:
            code.putln('%s.%s = *%s;' % (type.vtable_cname, Naming.obj_base_cname, type.base_type.vtabptr_cname))
        c_method_entries = [entry for entry in type.scope.cfunc_entries if entry.func_cname]
        if c_method_entries:
            for meth_entry in c_method_entries:
                vtable_type = meth_entry.vtable_type or meth_entry.type
                cast = vtable_type.signature_cast_string()
                code.putln('%s.%s = %s%s;' % (type.vtable_cname, meth_entry.cname, cast, meth_entry.func_cname))