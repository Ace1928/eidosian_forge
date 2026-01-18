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
def c_const_or_volatile_type(base_type, is_const, is_volatile):
    return _construct_type_from_base(CConstOrVolatileType, base_type, is_const, is_volatile)