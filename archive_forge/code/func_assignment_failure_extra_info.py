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
def assignment_failure_extra_info(self, src_type, src_name):
    if self.base_type.is_cfunction and src_type.is_ptr:
        src_type = src_type.base_type.resolve()
    if self.base_type.is_cfunction and src_type.is_cfunction:
        copied_src_type = copy.copy(src_type)
        copied_src_type.exception_check = self.base_type.exception_check
        copied_src_type.exception_value = self.base_type.exception_value
        if self.base_type.pointer_assignable_from_resolved_type(copied_src_type):
            msg = 'Exception values are incompatible.'
            if not self.base_type.exception_check and (not self.base_type.exception_value):
                if src_name is None:
                    src_name = 'the value being assigned'
                else:
                    src_name = "'{}'".format(src_name)
                msg += " Suggest adding 'noexcept' to the type of {0}.".format(src_name)
            return msg
    return super(CPtrType, self).assignment_failure_extra_info(src_type, src_name)