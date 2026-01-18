from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def lookup_operator(self, operator, operands):
    if operands[0].type.is_cpp_class:
        obj_type = operands[0].type
        method = obj_type.scope.lookup('operator%s' % operator)
        if method is not None:
            arg_types = [arg.type for arg in operands[1:]]
            res = PyrexTypes.best_match(arg_types, method.all_alternatives())
            if res is not None:
                return res
    function = self.lookup('operator%s' % operator)
    function_alternatives = []
    if function is not None:
        function_alternatives = function.all_alternatives()
    method_alternatives = []
    if len(operands) == 2:
        for n in range(2):
            if operands[n].type.is_cpp_class:
                obj_type = operands[n].type
                method = obj_type.scope.lookup('operator%s' % operator)
                if method is not None:
                    method_alternatives += method.all_alternatives()
    if not method_alternatives and (not function_alternatives):
        return None
    all_alternatives = list(set(method_alternatives + function_alternatives))
    return PyrexTypes.best_match([arg.type for arg in operands], all_alternatives)