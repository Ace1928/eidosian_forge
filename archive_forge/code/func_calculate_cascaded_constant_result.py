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
def calculate_cascaded_constant_result(self, operand1_result):
    func = compile_time_binary_operators[self.operator]
    operand2_result = self.operand2.constant_result
    if isinstance(operand1_result, any_string_type) and isinstance(operand2_result, any_string_type) and (type(operand1_result) != type(operand2_result)):
        return
    if self.operator in ('in', 'not_in'):
        if isinstance(self.operand2, (ListNode, TupleNode, SetNode)):
            if not self.operand2.args:
                self.constant_result = self.operator == 'not_in'
                return
            elif isinstance(self.operand2, ListNode) and (not self.cascade):
                self.operand2 = self.operand2.as_tuple()
        elif isinstance(self.operand2, DictNode):
            if not self.operand2.key_value_pairs:
                self.constant_result = self.operator == 'not_in'
                return
    self.constant_result = func(operand1_result, operand2_result)