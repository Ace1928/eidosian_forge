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
def spanning_type(type1, type2):
    if type1 == type2:
        return type1
    elif type1 is py_object_type or type2 is py_object_type:
        return py_object_type
    elif type1 is c_py_unicode_type or type2 is c_py_unicode_type:
        return py_object_type
    span_type = _spanning_type(type1, type2)
    if span_type is None:
        return py_object_type
    return span_type