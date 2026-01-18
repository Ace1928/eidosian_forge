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
def extra_index_params(self, code):
    if self.index.type.is_int:
        is_list = self.base.type is list_type
        wraparound = bool(code.globalstate.directives['wraparound']) and self.original_index_type.signed and (not (isinstance(self.index.constant_result, _py_int_types) and self.index.constant_result >= 0))
        boundscheck = bool(code.globalstate.directives['boundscheck'])
        return ', %s, %d, %s, %d, %d, %d' % (self.original_index_type.empty_declaration_code(), self.original_index_type.signed and 1 or 0, self.original_index_type.to_py_function, is_list, wraparound, boundscheck)
    else:
        return ''