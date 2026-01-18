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
def calculate_access_code(self):
    obj = self.obj
    obj_code = obj.result_as(obj.type)
    if self.entry and self.entry.is_cmethod:
        if obj.type.is_extension_type and (not self.entry.is_builtin_cmethod):
            if self.entry.final_func_cname:
                return self.entry.final_func_cname
            if self.type.from_fused:
                self.member = self.entry.cname
            return '((struct %s *)%s%s%s)->%s' % (obj.type.vtabstruct_cname, obj_code, self.op, obj.type.vtabslot_cname, self.member)
        elif self.result_is_used:
            return self.member
        return
    elif obj.type.is_complex:
        return '__Pyx_C%s(%s)' % (self.member.upper(), obj_code)
    else:
        if obj.type.is_builtin_type and self.entry and self.entry.is_variable:
            obj_code = obj.type.cast_code(obj.result(), to_object_struct=True)
        return '%s%s%s' % (obj_code, self.op, self.member)