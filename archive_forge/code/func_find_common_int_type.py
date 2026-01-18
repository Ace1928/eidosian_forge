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
def find_common_int_type(self, env, op, operand1, operand2):
    type1 = operand1.type
    type2 = operand2.type
    type1_can_be_int = False
    type2_can_be_int = False
    if operand1.is_string_literal and operand1.can_coerce_to_char_literal():
        type1_can_be_int = True
    if operand2.is_string_literal and operand2.can_coerce_to_char_literal():
        type2_can_be_int = True
    if type1.is_int:
        if type2_can_be_int:
            return type1
    elif type2.is_int:
        if type1_can_be_int:
            return type2
    elif type1_can_be_int:
        if type2_can_be_int:
            if Builtin.unicode_type in (type1, type2):
                return PyrexTypes.c_py_ucs4_type
            else:
                return PyrexTypes.c_uchar_type
    return None