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
def public_decl(base_code, dll_linkage):
    if dll_linkage:
        return '%s(%s)' % (dll_linkage, base_code.replace(',', ' __PYX_COMMA '))
    else:
        return base_code