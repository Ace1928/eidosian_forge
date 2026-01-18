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
def same_as(self, other_type, compare_base=True):
    if not other_type.is_buffer:
        return other_type.same_as(self.base)
    return self.dtype.same_as(other_type.dtype) and self.ndim == other_type.ndim and (self.mode == other_type.mode) and (self.cast == other_type.cast) and (not compare_base or self.base.same_as(other_type.base))