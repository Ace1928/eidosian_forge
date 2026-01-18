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
def create_typedef_type(name, base_type, cname, is_external=0, namespace=None):
    if is_external:
        if base_type.is_complex or base_type.is_fused:
            raise ValueError('%s external typedefs not supported' % ('Fused' if base_type.is_fused else 'Complex'))
    if base_type.is_complex or base_type.is_fused:
        return base_type
    return CTypedefType(name, base_type, cname, is_external, namespace)