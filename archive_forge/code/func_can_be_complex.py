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
def can_be_complex(self):
    fields = self.scope.var_entries
    if len(fields) != 2:
        return False
    a, b = fields
    return a.type.is_float and b.type.is_float and (a.type.empty_declaration_code() == b.type.empty_declaration_code())