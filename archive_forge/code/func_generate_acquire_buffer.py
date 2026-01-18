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
def generate_acquire_buffer(self, rhs, code):
    pretty_rhs = isinstance(rhs, NameNode) or rhs.is_temp
    if pretty_rhs:
        rhstmp = rhs.result_as(self.ctype())
    else:
        rhstmp = code.funcstate.allocate_temp(self.entry.type, manage_ref=False)
        code.putln('%s = %s;' % (rhstmp, rhs.result_as(self.ctype())))
    from . import Buffer
    Buffer.put_assign_to_buffer(self.result(), rhstmp, self.entry, is_initialized=not self.lhs_of_first_assignment, pos=self.pos, code=code)
    if not pretty_rhs:
        code.putln('%s = 0;' % rhstmp)
        code.funcstate.release_temp(rhstmp)