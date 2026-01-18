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
def analyse_cpp_comparison(self, env):
    type1 = self.operand1.type
    type2 = self.operand2.type
    self.is_pycmp = False
    entry = env.lookup_operator(self.operator, [self.operand1, self.operand2])
    if entry is None:
        error(self.pos, "Invalid types for '%s' (%s, %s)" % (self.operator, type1, type2))
        self.type = PyrexTypes.error_type
        self.result_code = '<error>'
        return
    func_type = entry.type
    if func_type.is_ptr:
        func_type = func_type.base_type
    self.exception_check = func_type.exception_check
    self.exception_value = func_type.exception_value
    if self.exception_check == '+':
        self.is_temp = True
        if needs_cpp_exception_conversion(self):
            env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
    if len(func_type.args) == 1:
        self.operand2 = self.operand2.coerce_to(func_type.args[0].type, env)
    else:
        self.operand1 = self.operand1.coerce_to(func_type.args[0].type, env)
        self.operand2 = self.operand2.coerce_to(func_type.args[1].type, env)
    self.type = func_type.return_type