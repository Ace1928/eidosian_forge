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
def same_calling_convention_as(self, other):
    sc1 = self.calling_convention == '__stdcall'
    sc2 = other.calling_convention == '__stdcall'
    return sc1 == sc2