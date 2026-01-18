from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def generate_cyfunction_code(self, code):
    if self.specialized_cpdefs:
        def_node = self.specialized_cpdefs[0]
    else:
        def_node = self.def_node
    if self.specialized_cpdefs or self.is_specialization:
        code.globalstate.use_utility_code(UtilityCode.load_cached('FusedFunction', 'CythonFunction.c'))
        constructor = '__pyx_FusedFunction_New'
    else:
        code.globalstate.use_utility_code(UtilityCode.load_cached('CythonFunction', 'CythonFunction.c'))
        constructor = '__Pyx_CyFunction_New'
    if self.code_object:
        code_object_result = self.code_object.py_result()
    else:
        code_object_result = 'NULL'
    flags = []
    if def_node.is_staticmethod:
        flags.append('__Pyx_CYFUNCTION_STATICMETHOD')
    elif def_node.is_classmethod:
        flags.append('__Pyx_CYFUNCTION_CLASSMETHOD')
    if def_node.local_scope.parent_scope.is_c_class_scope and (not def_node.entry.is_anonymous):
        flags.append('__Pyx_CYFUNCTION_CCLASS')
    if def_node.is_coroutine:
        flags.append('__Pyx_CYFUNCTION_COROUTINE')
    if flags:
        flags = ' | '.join(flags)
    else:
        flags = '0'
    code.putln('%s = %s(&%s, %s, %s, %s, %s, %s, %s); %s' % (self.result(), constructor, self.pymethdef_cname, flags, self.get_py_qualified_name(code), self.closure_result_code(), self.get_py_mod_name(code), Naming.moddict_cname, code_object_result, code.error_goto_if_null(self.result(), self.pos)))
    self.generate_gotref(code)
    if def_node.requires_classobj:
        assert code.pyclass_stack, 'pyclass_stack is empty'
        class_node = code.pyclass_stack[-1]
        code.put_incref(self.py_result(), py_object_type)
        code.putln('PyList_Append(%s, %s);' % (class_node.class_cell.result(), self.result()))
        self.generate_giveref(code)
    if self.defaults:
        code.putln('if (!__Pyx_CyFunction_InitDefaults(%s, sizeof(%s), %d)) %s' % (self.result(), self.defaults_struct.name, self.defaults_pyobjects, code.error_goto(self.pos)))
        defaults = '__Pyx_CyFunction_Defaults(%s, %s)' % (self.defaults_struct.name, self.result())
        for arg, entry in self.defaults:
            arg.generate_assignment_code(code, target='%s->%s' % (defaults, entry.cname))
    if self.defaults_tuple:
        code.putln('__Pyx_CyFunction_SetDefaultsTuple(%s, %s);' % (self.result(), self.defaults_tuple.py_result()))
    if not self.specialized_cpdefs:
        if self.defaults_kwdict:
            code.putln('__Pyx_CyFunction_SetDefaultsKwDict(%s, %s);' % (self.result(), self.defaults_kwdict.py_result()))
        if def_node.defaults_getter:
            code.putln('__Pyx_CyFunction_SetDefaultsGetter(%s, %s);' % (self.result(), def_node.defaults_getter.entry.pyfunc_cname))
        if self.annotations_dict:
            code.putln('__Pyx_CyFunction_SetAnnotationsDict(%s, %s);' % (self.result(), self.annotations_dict.py_result()))