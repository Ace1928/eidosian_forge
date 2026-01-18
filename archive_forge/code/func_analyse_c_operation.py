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
def analyse_c_operation(self, env):
    NumBinopNode.analyse_c_operation(self, env)
    if self.type.is_complex:
        if self.type.real_type.is_float:
            self.operand1 = self.operand1.coerce_to(self.type, env)
            self.operand2 = self.operand2.coerce_to(self.type, env)
            self.pow_func = self.type.binary_op('**')
        else:
            error(self.pos, 'complex int powers not supported')
            self.pow_func = '<error>'
    elif self.type.is_float:
        self.pow_func = 'pow' + self.type.math_h_modifier
    elif self.type.is_int:
        self.pow_func = '__Pyx_pow_%s' % self.type.empty_declaration_code().replace(' ', '_')
        env.use_utility_code(UtilityCode.load_cached('IntPow', 'CMath.c').specialize(func_name=self.pow_func, type=self.type.empty_declaration_code(), signed=self.type.signed and 1 or 0))
    elif not self.type.is_error:
        error(self.pos, 'got unexpected types for C power operator: %s, %s' % (self.operand1.type, self.operand2.type))