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
def lazy_utility_callback(code):
    context['dtype_typeinfo'] = Buffer.get_type_information_cname(code, self.dtype)
    return TempitaUtilityCode.load('ObjectToMemviewSlice', 'MemoryView_C.c', context=context)