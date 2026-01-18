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
def from_py_call_code(self, source_code, result_code, error_pos, code, from_py_function=None, error_condition=None, special_none_cvalue=None):
    assert not error_condition, '%s: %s' % (error_pos, error_condition)
    assert not special_none_cvalue, '%s: %s' % (error_pos, special_none_cvalue)
    call_code = '%s(%s, %s, %s)' % (from_py_function or self.from_py_function, source_code, result_code, self.size)
    return code.error_goto_if_neg(call_code, error_pos)