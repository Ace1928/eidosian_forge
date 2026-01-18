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
def generate_objstruct_definition(self, type, code):
    code.mark_pos(type.pos)
    if not type.scope:
        return
    header, footer = self.sue_header_footer(type, 'struct', type.objstruct_cname)
    code.putln(header)
    base_type = type.base_type
    if base_type:
        basestruct_cname = base_type.objstruct_cname
        if basestruct_cname == 'PyTypeObject':
            basestruct_cname = 'PyHeapTypeObject'
        code.putln('%s%s %s;' % (('struct ', '')[base_type.typedef_flag], basestruct_cname, Naming.obj_base_cname))
    else:
        code.putln('PyObject_HEAD')
    if type.vtabslot_cname and (not (type.base_type and type.base_type.vtabslot_cname)):
        code.putln('struct %s *%s;' % (type.vtabstruct_cname, type.vtabslot_cname))
    for attr in type.scope.var_entries:
        if attr.is_declared_generic:
            attr_type = py_object_type
        else:
            attr_type = attr.type
        if attr.is_cpp_optional:
            decl = attr_type.cpp_optional_declaration_code(attr.cname)
        else:
            decl = attr_type.declaration_code(attr.cname)
        type.scope.use_entry_utility_code(attr)
        code.putln('%s;' % decl)
    code.putln(footer)
    if type.objtypedef_cname is not None:
        code.putln('typedef struct %s %s;' % (type.objstruct_cname, type.objtypedef_cname))