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
def find_suitable_type_for_value(self):
    if self.constant_result is constant_value_not_set:
        try:
            self.calculate_constant_result()
        except ValueError:
            pass
    if self.is_c_literal or not self.has_constant_result() or self.unsigned or (self.longness == 'LL'):
        rank = self.longness == 'LL' and 2 or 1
        suitable_type = PyrexTypes.modifiers_and_name_to_type[not self.unsigned, rank, 'int']
        if self.type:
            suitable_type = PyrexTypes.widest_numeric_type(suitable_type, self.type)
    elif -2 ** 31 <= self.constant_result < 2 ** 31:
        if self.type and self.type.is_int:
            suitable_type = self.type
        else:
            suitable_type = PyrexTypes.c_long_type
    else:
        suitable_type = PyrexTypes.py_object_type
    return suitable_type