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
def default_coerced_ctype(self):
    if self.name in ('bytes', 'bytearray'):
        return c_char_ptr_type
    elif self.name == 'bool':
        return c_bint_type
    elif self.name == 'float':
        return c_double_type
    return None