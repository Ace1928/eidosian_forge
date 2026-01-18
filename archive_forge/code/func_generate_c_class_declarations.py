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
def generate_c_class_declarations(self, env, code, definition, globalstate):
    module_state = globalstate['module_state']
    module_state_defines = globalstate['module_state_defines']
    module_state_clear = globalstate['module_state_clear']
    module_state_traverse = globalstate['module_state_traverse']
    module_state_typeobj = module_state.insertion_point()
    module_state_defines_typeobj = module_state_defines.insertion_point()
    for writer in [module_state_typeobj, module_state_defines_typeobj]:
        writer.putln('#if CYTHON_USE_MODULE_STATE')
    for entry in env.c_class_entries:
        if definition or entry.defined_in_pxd:
            module_state.putln('PyTypeObject *%s;' % entry.type.typeptr_cname)
            module_state_defines.putln('#define %s %s->%s' % (entry.type.typeptr_cname, Naming.modulestateglobal_cname, entry.type.typeptr_cname))
            module_state_clear.putln('Py_CLEAR(clear_module_state->%s);' % entry.type.typeptr_cname)
            module_state_traverse.putln('Py_VISIT(traverse_module_state->%s);' % entry.type.typeptr_cname)
            if entry.type.typeobj_cname is not None:
                module_state_typeobj.putln('PyObject *%s;' % entry.type.typeobj_cname)
                module_state_defines_typeobj.putln('#define %s %s->%s' % (entry.type.typeobj_cname, Naming.modulestateglobal_cname, entry.type.typeobj_cname))
                module_state_clear.putln('Py_CLEAR(clear_module_state->%s);' % entry.type.typeobj_cname)
                module_state_traverse.putln('Py_VISIT(traverse_module_state->%s);' % entry.type.typeobj_cname)
    for writer in [module_state_typeobj, module_state_defines_typeobj]:
        writer.putln('#endif')