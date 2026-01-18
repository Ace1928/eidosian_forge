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
def set_py_result_type(self, function, func_type=None):
    if func_type is None:
        func_type = function.type
    if func_type is Builtin.type_type and (function.is_name and function.entry and function.entry.is_builtin and (function.entry.name in Builtin.types_that_construct_their_instance)):
        if function.entry.name == 'float':
            self.type = PyrexTypes.c_double_type
            self.result_ctype = PyrexTypes.c_double_type
        else:
            self.type = Builtin.builtin_types[function.entry.name]
            self.result_ctype = py_object_type
        self.may_return_none = False
    elif function.is_name and function.type_entry:
        self.type = function.type_entry.type
        self.result_ctype = py_object_type
        self.may_return_none = False
    else:
        self.type = py_object_type