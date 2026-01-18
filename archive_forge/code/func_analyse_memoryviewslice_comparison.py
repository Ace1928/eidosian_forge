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
def analyse_memoryviewslice_comparison(self, env):
    have_none = self.operand1.is_none or self.operand2.is_none
    have_slice = self.operand1.type.is_memoryviewslice or self.operand2.type.is_memoryviewslice
    ops = ('==', '!=', 'is', 'is_not')
    if have_slice and have_none and (self.operator in ops):
        self.is_pycmp = False
        self.type = PyrexTypes.c_bint_type
        self.is_memslice_nonecheck = True
        return True
    return False