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
def compute_c_result_type(self, type1, type2):
    from numbers import Real
    c_result_type = None
    op1_is_definitely_positive = self.operand1.has_constant_result() and self.operand1.constant_result >= 0 or (type1.is_int and type1.signed == 0)
    type2_is_int = type2.is_int or (self.operand2.has_constant_result() and isinstance(self.operand2.constant_result, Real) and (int(self.operand2.constant_result) == self.operand2.constant_result))
    needs_widening = False
    if self.is_cpow:
        c_result_type = super(PowNode, self).compute_c_result_type(type1, type2)
        if not self.operand2.has_constant_result():
            needs_widening = isinstance(self.operand2.constant_result, _py_int_types) and self.operand2.constant_result < 0
    elif op1_is_definitely_positive or type2_is_int:
        c_result_type = super(PowNode, self).compute_c_result_type(type1, type2)
        if not self.operand2.has_constant_result():
            needs_widening = type2.is_int and type2.signed
            if needs_widening:
                self.type_was_inferred = True
        else:
            needs_widening = isinstance(self.operand2.constant_result, _py_int_types) and self.operand2.constant_result < 0
    elif self.c_types_okay(type1, type2):
        c_result_type = PyrexTypes.soft_complex_type
        self.type_was_inferred = True
    if needs_widening:
        c_result_type = PyrexTypes.widest_numeric_type(c_result_type, PyrexTypes.c_double_type)
    return c_result_type