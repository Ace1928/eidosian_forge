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
def sign_and_name(self):
    real_type_name = self.real_type.specialization_name()
    real_type_name = real_type_name.replace('long__double', 'long_double')
    real_type_name = real_type_name.replace('PY_LONG_LONG', 'long_long')
    return Naming.type_prefix + real_type_name + '_complex'