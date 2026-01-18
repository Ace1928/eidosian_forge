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
def remove_cv_ref(tp, remove_fakeref=False):
    last_tp = None
    while tp != last_tp:
        last_tp = tp
        if tp.is_cv_qualified:
            tp = tp.cv_base_type
        if tp.is_reference and (not tp.is_fake_reference or remove_fakeref):
            tp = tp.ref_base_type
    return tp