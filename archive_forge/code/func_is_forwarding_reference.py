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
def is_forwarding_reference(self):
    if self.type.is_rvalue_reference:
        if isinstance(self.type.ref_base_type, TemplatePlaceholderType) and (not self.type.ref_base_type.is_cv_qualified):
            return True
    return False