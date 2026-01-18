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
def assignable_from(self, src_type):
    if not src_type.is_complex and src_type.is_numeric and src_type.is_typedef and src_type.typedef_is_external:
        return False
    elif src_type.is_pyobject:
        return True
    else:
        return super(CComplexType, self).assignable_from(src_type)