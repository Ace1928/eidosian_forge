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
def generate_buffer_setitem_code(self, rhs, code, op=''):
    base_type = self.base.type
    if is_pythran_expr(base_type) and is_pythran_supported_type(rhs.type):
        obj = code.funcstate.allocate_temp(PythranExpr(pythran_type(self.base.type)), manage_ref=False)
        code.putln('__Pyx_call_destructor(%s);' % obj)
        code.putln('new (&%s) decltype(%s){%s};' % (obj, obj, self.base.pythran_result()))
        code.putln('%s%s %s= %s;' % (obj, pythran_indexing_code(self.indices), op, rhs.pythran_result()))
        code.funcstate.release_temp(obj)
        return
    buffer_entry, ptrexpr = self.buffer_lookup_code(code)
    if self.buffer_type.dtype.is_pyobject:
        ptr = code.funcstate.allocate_temp(buffer_entry.buf_ptr_type, manage_ref=False)
        rhs_code = rhs.result()
        code.putln('%s = %s;' % (ptr, ptrexpr))
        code.put_xgotref('*%s' % ptr, self.buffer_type.dtype)
        code.putln('__Pyx_INCREF(%s); __Pyx_XDECREF(*%s);' % (rhs_code, ptr))
        code.putln('*%s %s= %s;' % (ptr, op, rhs_code))
        code.put_xgiveref('*%s' % ptr, self.buffer_type.dtype)
        code.funcstate.release_temp(ptr)
    else:
        code.putln('*%s %s= %s;' % (ptrexpr, op, rhs.result()))