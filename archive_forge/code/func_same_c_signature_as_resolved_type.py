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
def same_c_signature_as_resolved_type(self, other_type, as_cmethod=False, as_pxd_definition=False, exact_semantics=True):
    if other_type is error_type:
        return 1
    if not other_type.is_cfunction:
        return 0
    if self.is_overridable != other_type.is_overridable:
        return 0
    nargs = len(self.args)
    if nargs != len(other_type.args):
        return 0
    for i in range(as_cmethod, nargs):
        if not self.args[i].type.same_as(other_type.args[i].type):
            return 0
    if self.has_varargs != other_type.has_varargs:
        return 0
    if self.optional_arg_count != other_type.optional_arg_count:
        return 0
    if as_pxd_definition:
        if not self.return_type.subtype_of_resolved_type(other_type.return_type):
            return 0
    elif not self.return_type.same_as(other_type.return_type):
        return 0
    if not self.same_calling_convention_as(other_type):
        return 0
    if exact_semantics:
        if self.exception_check != other_type.exception_check:
            return 0
        if not self._same_exception_value(other_type.exception_value):
            return 0
    elif not self._is_exception_compatible_with(other_type):
        return 0
    return 1