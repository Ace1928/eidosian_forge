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
def assert_direct_dims(self, pos):
    for access, packing in self.axes:
        if access != 'direct':
            error(pos, 'All dimensions must be direct')
            return False
    return True