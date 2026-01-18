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
def c_call_code(self):
    func_type = self.function_type()
    if self.type is PyrexTypes.error_type or not func_type.is_cfunction:
        return '<error>'
    formal_args = func_type.args
    arg_list_code = []
    args = list(zip(formal_args, self.args))
    max_nargs = len(func_type.args)
    expected_nargs = max_nargs - func_type.optional_arg_count
    actual_nargs = len(self.args)
    for formal_arg, actual_arg in args[:expected_nargs]:
        arg_code = actual_arg.move_result_rhs_as(formal_arg.type)
        arg_list_code.append(arg_code)
    if func_type.is_overridable:
        arg_list_code.append(str(int(self.wrapper_call or self.function.entry.is_unbound_cmethod)))
    if func_type.optional_arg_count:
        if expected_nargs == actual_nargs:
            optional_args = 'NULL'
        else:
            optional_args = '&%s' % self.opt_arg_struct
        arg_list_code.append(optional_args)
    for actual_arg in self.args[len(formal_args):]:
        arg_list_code.append(actual_arg.move_result_rhs())
    result = '%s(%s)' % (self.function.result(), ', '.join(arg_list_code))
    return result