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
def generate_typeobj_definition(self, modname, entry, code):
    type = entry.type
    scope = type.scope
    for suite in TypeSlots.get_slot_table(code.globalstate.directives).substructures:
        suite.generate_substructure(scope, code)
    code.putln('')
    if entry.visibility == 'public':
        header = 'DL_EXPORT(PyTypeObject) %s = {'
    else:
        header = 'static PyTypeObject %s = {'
    code.putln(header % type.typeobj_cname)
    code.putln('PyVarObject_HEAD_INIT(0, 0)')
    classname = scope.class_name.as_c_string_literal()
    code.putln('"%s."%s, /*tp_name*/' % (self.full_module_name, classname))
    if type.typedef_flag:
        objstruct = type.objstruct_cname
    else:
        objstruct = 'struct %s' % type.objstruct_cname
    code.putln('sizeof(%s), /*tp_basicsize*/' % objstruct)
    code.putln('0, /*tp_itemsize*/')
    for slot in TypeSlots.get_slot_table(code.globalstate.directives):
        slot.generate(scope, code)
    code.putln('};')