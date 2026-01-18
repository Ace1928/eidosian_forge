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
def generate_starred_assignment_code(self, rhs, code):
    for i, arg in enumerate(self.args):
        if arg.is_starred:
            starred_target = self.unpacked_items[i]
            unpacked_fixed_items_left = self.unpacked_items[:i]
            unpacked_fixed_items_right = self.unpacked_items[i + 1:]
            break
    else:
        assert False
    iterator_temp = None
    if unpacked_fixed_items_left:
        for item in unpacked_fixed_items_left:
            item.allocate(code)
        code.putln('{')
        iterator_temp = self.generate_generic_parallel_unpacking_code(code, rhs, unpacked_fixed_items_left, use_loop=True, terminate=False)
        for i, item in enumerate(unpacked_fixed_items_left):
            value_node = self.coerced_unpacked_items[i]
            value_node.generate_evaluation_code(code)
        code.putln('}')
    starred_target.allocate(code)
    target_list = starred_target.result()
    code.putln('%s = %s(%s); %s' % (target_list, '__Pyx_PySequence_ListKeepNew' if not iterator_temp and rhs.is_temp and (rhs.type in (py_object_type, list_type)) else 'PySequence_List', iterator_temp or rhs.py_result(), code.error_goto_if_null(target_list, self.pos)))
    starred_target.generate_gotref(code)
    if iterator_temp:
        code.put_decref_clear(iterator_temp, py_object_type)
        code.funcstate.release_temp(iterator_temp)
    else:
        rhs.generate_disposal_code(code)
    if unpacked_fixed_items_right:
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
        length_temp = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
        code.putln('%s = PyList_GET_SIZE(%s);' % (length_temp, target_list))
        code.putln('if (unlikely(%s < %d)) {' % (length_temp, len(unpacked_fixed_items_right)))
        code.putln('__Pyx_RaiseNeedMoreValuesError(%d+%s); %s' % (len(unpacked_fixed_items_left), length_temp, code.error_goto(self.pos)))
        code.putln('}')
        for item in unpacked_fixed_items_right[::-1]:
            item.allocate(code)
        for i, (item, coerced_arg) in enumerate(zip(unpacked_fixed_items_right[::-1], self.coerced_unpacked_items[::-1])):
            code.putln('#if CYTHON_COMPILING_IN_CPYTHON')
            code.putln('%s = PyList_GET_ITEM(%s, %s-%d); ' % (item.py_result(), target_list, length_temp, i + 1))
            code.putln('((PyVarObject*)%s)->ob_size--;' % target_list)
            code.putln('#else')
            code.putln('%s = PySequence_ITEM(%s, %s-%d); ' % (item.py_result(), target_list, length_temp, i + 1))
            code.putln('#endif')
            item.generate_gotref(code)
            coerced_arg.generate_evaluation_code(code)
        code.putln('#if !CYTHON_COMPILING_IN_CPYTHON')
        sublist_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        code.putln('%s = PySequence_GetSlice(%s, 0, %s-%d); %s' % (sublist_temp, target_list, length_temp, len(unpacked_fixed_items_right), code.error_goto_if_null(sublist_temp, self.pos)))
        code.put_gotref(sublist_temp, py_object_type)
        code.funcstate.release_temp(length_temp)
        code.put_decref(target_list, py_object_type)
        code.putln('%s = %s; %s = NULL;' % (target_list, sublist_temp, sublist_temp))
        code.putln('#else')
        code.putln('CYTHON_UNUSED_VAR(%s);' % sublist_temp)
        code.funcstate.release_temp(sublist_temp)
        code.putln('#endif')
    for i, arg in enumerate(self.args):
        arg.generate_assignment_code(self.coerced_unpacked_items[i], code)