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
def c_tuple_type(components):
    components = tuple(components)
    cname = Naming.ctuple_type_prefix + type_list_identifier(components)
    tuple_type = CTupleType(cname, components)
    return tuple_type