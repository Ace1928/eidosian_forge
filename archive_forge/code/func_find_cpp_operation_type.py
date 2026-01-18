from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def find_cpp_operation_type(self, operator, operand_type=None):
    operands = [self]
    if operand_type is not None:
        operands.append(operand_type)
    operator_entry = self.scope.lookup_operator_for_types(None, operator, operands)
    if not operator_entry:
        return None
    func_type = operator_entry.type
    if func_type.is_ptr:
        func_type = func_type.base_type
    return func_type.return_type