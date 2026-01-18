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
def cascaded_compile_time_value(self, operand1, denv):
    func = get_compile_time_binop(self)
    operand2 = self.operand2.compile_time_value(denv)
    try:
        result = func(operand1, operand2)
    except Exception as e:
        self.compile_time_value_error(e)
        result = None
    if result:
        cascade = self.cascade
        if cascade:
            result = result and cascade.cascaded_compile_time_value(operand2, denv)
    return result