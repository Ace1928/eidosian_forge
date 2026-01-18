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
def py_operation_function(self, code):
    if self.type.is_pyobject and self.operand1.constant_result == 2 and isinstance(self.operand1.constant_result, _py_int_types) and (self.operand2.type is py_object_type):
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyNumberPow2', 'Optimize.c'))
        if self.inplace:
            return '__Pyx_PyNumber_InPlacePowerOf2'
        else:
            return '__Pyx_PyNumber_PowerOf2'
    return super(PowNode, self).py_operation_function(code)