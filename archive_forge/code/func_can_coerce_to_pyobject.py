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
def can_coerce_to_pyobject(self, env):
    for component in self.components:
        if not component.can_coerce_to_pyobject(env):
            return False
    return True