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
def function_header_code(self, func_name, arg_code):
    if self.is_const_method:
        trailer = ' const'
    else:
        trailer = ''
    return '%s%s(%s)%s' % (self.calling_convention_prefix(), func_name, arg_code, trailer)