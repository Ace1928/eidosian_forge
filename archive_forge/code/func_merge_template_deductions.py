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
def merge_template_deductions(a, b):
    if a is None or b is None:
        return None
    all = a
    for param, value in b.items():
        if param in all:
            if a[param] != b[param]:
                return None
        else:
            all[param] = value
    return all