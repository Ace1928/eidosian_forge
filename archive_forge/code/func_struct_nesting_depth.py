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
def struct_nesting_depth(self):
    child_depths = [x.type.struct_nesting_depth() for x in self.scope.var_entries]
    return max(child_depths) + 1