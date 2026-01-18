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
def axes_to_name(self):
    """Return an abbreviated name for our axes"""
    from . import MemoryView
    d = MemoryView._spec_to_abbrev
    return ''.join(['%s%s' % (d[a], d[p]) for a, p in self.axes])