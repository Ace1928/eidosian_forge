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
def parse_basic_type(name):
    base = None
    if name.startswith('p_'):
        base = parse_basic_type(name[2:])
    elif name.startswith('p'):
        base = parse_basic_type(name[1:])
    elif name.endswith('*'):
        base = parse_basic_type(name[:-1])
    if base:
        return CPtrType(base)
    basic_type = simple_c_type(1, 0, name)
    if basic_type:
        return basic_type
    signed = 1
    longness = 0
    if name == 'Py_UNICODE':
        signed = 0
    elif name == 'Py_UCS4':
        signed = 0
    elif name == 'Py_hash_t':
        signed = 2
    elif name == 'Py_ssize_t':
        signed = 2
    elif name == 'ssize_t':
        signed = 2
    elif name == 'size_t':
        signed = 0
    elif name == 'ptrdiff_t':
        signed = 2
    else:
        if name.startswith('u'):
            name = name[1:]
            signed = 0
        elif name.startswith('s') and (not name.startswith('short')):
            name = name[1:]
            signed = 2
        longness = 0
        while name.startswith('short'):
            name = name.replace('short', '', 1).strip()
            longness -= 1
        while name.startswith('long'):
            name = name.replace('long', '', 1).strip()
            longness += 1
        if longness != 0 and (not name):
            name = 'int'
    return simple_c_type(signed, longness, name)