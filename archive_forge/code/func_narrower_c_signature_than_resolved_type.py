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
def narrower_c_signature_than_resolved_type(self, other_type, as_cmethod):
    if other_type is error_type:
        return 1
    if not other_type.is_cfunction:
        return 0
    nargs = len(self.args)
    if nargs != len(other_type.args):
        return 0
    for i in range(as_cmethod, nargs):
        if not self.args[i].type.subtype_of_resolved_type(other_type.args[i].type):
            return 0
        else:
            self.args[i].needs_type_test = other_type.args[i].needs_type_test or not self.args[i].type.same_as(other_type.args[i].type)
    if self.has_varargs != other_type.has_varargs:
        return 0
    if self.optional_arg_count != other_type.optional_arg_count:
        return 0
    if not self.return_type.subtype_of_resolved_type(other_type.return_type):
        return 0
    if not self.exception_check and other_type.exception_check:
        return 0
    if not self._same_exception_value(other_type.exception_value):
        return 0
    return 1