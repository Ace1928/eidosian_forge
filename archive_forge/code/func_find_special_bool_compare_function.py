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
def find_special_bool_compare_function(self, env, operand1, result_is_bool=False):
    if self.operator in ('==', '!='):
        type1, type2 = (operand1.type, self.operand2.type)
        if result_is_bool or (type1.is_builtin_type and type2.is_builtin_type):
            if type1 is Builtin.unicode_type or type2 is Builtin.unicode_type:
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('UnicodeEquals', 'StringTools.c')
                self.special_bool_cmp_function = '__Pyx_PyUnicode_Equals'
                return True
            elif type1 is Builtin.bytes_type or type2 is Builtin.bytes_type:
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('BytesEquals', 'StringTools.c')
                self.special_bool_cmp_function = '__Pyx_PyBytes_Equals'
                return True
            elif type1 is Builtin.basestring_type or type2 is Builtin.basestring_type:
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('UnicodeEquals', 'StringTools.c')
                self.special_bool_cmp_function = '__Pyx_PyUnicode_Equals'
                return True
            elif type1 is Builtin.str_type or type2 is Builtin.str_type:
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('StrEquals', 'StringTools.c')
                self.special_bool_cmp_function = '__Pyx_PyString_Equals'
                return True
            elif result_is_bool:
                from .Optimize import optimise_numeric_binop
                result = optimise_numeric_binop('Eq' if self.operator == '==' else 'Ne', self, PyrexTypes.c_bint_type, operand1, self.operand2)
                if result:
                    self.special_bool_cmp_function, self.special_bool_cmp_utility_code, self.special_bool_extra_args, _ = result
                    return True
    elif self.operator in ('in', 'not_in'):
        if self.operand2.type is Builtin.dict_type:
            self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
            self.special_bool_cmp_utility_code = UtilityCode.load_cached('PyDictContains', 'ObjectHandling.c')
            self.special_bool_cmp_function = '__Pyx_PyDict_ContainsTF'
            return True
        elif self.operand2.type is Builtin.set_type:
            self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
            self.special_bool_cmp_utility_code = UtilityCode.load_cached('PySetContains', 'ObjectHandling.c')
            self.special_bool_cmp_function = '__Pyx_PySet_ContainsTF'
            return True
        elif self.operand2.type is Builtin.unicode_type:
            self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
            self.special_bool_cmp_utility_code = UtilityCode.load_cached('PyUnicodeContains', 'StringTools.c')
            self.special_bool_cmp_function = '__Pyx_PyUnicode_ContainsTF'
            return True
        else:
            if not self.operand2.type.is_pyobject:
                self.operand2 = self.operand2.coerce_to_pyobject(env)
            self.special_bool_cmp_utility_code = UtilityCode.load_cached('PySequenceContains', 'ObjectHandling.c')
            self.special_bool_cmp_function = '__Pyx_PySequence_ContainsTF'
            return True
    return False