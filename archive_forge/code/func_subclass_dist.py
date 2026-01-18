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
def subclass_dist(self, super_type):
    if self.same_as_resolved_type(super_type):
        return 0
    elif not self.base_classes:
        return float('inf')
    else:
        return 1 + min((b.subclass_dist(super_type) for b in self.base_classes))