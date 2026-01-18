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
def generate_parallel_assignment_code(self, rhs, code):
    for item in self.unpacked_items:
        item.allocate(code)
    special_unpack = rhs.type is py_object_type or rhs.type in (tuple_type, list_type) or (not rhs.type.is_builtin_type)
    long_enough_for_a_loop = len(self.unpacked_items) > 3
    if special_unpack:
        self.generate_special_parallel_unpacking_code(code, rhs, use_loop=long_enough_for_a_loop)
    else:
        code.putln('{')
        self.generate_generic_parallel_unpacking_code(code, rhs, self.unpacked_items, use_loop=long_enough_for_a_loop)
        code.putln('}')
    for value_node in self.coerced_unpacked_items:
        value_node.generate_evaluation_code(code)
    for i in range(len(self.args)):
        self.args[i].generate_assignment_code(self.coerced_unpacked_items[i], code)