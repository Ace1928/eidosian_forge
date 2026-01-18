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
def is_reversed_cpp_iteration(self):
    """
        Returns True if the 'reversed' function is applied to a C++ iterable.

        This supports C++ classes with reverse_iterator implemented.
        """
    if not (isinstance(self.sequence, SimpleCallNode) and self.sequence.arg_tuple and (len(self.sequence.arg_tuple.args) == 1)):
        return False
    func = self.sequence.function
    if func.is_name and func.name == 'reversed':
        if not func.entry.is_builtin:
            return False
        arg = self.sequence.arg_tuple.args[0]
        if isinstance(arg, CoercionNode) and arg.arg.is_name:
            arg = arg.arg.entry
            return arg.type.is_cpp_class
    return False