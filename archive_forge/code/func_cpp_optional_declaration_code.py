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
def cpp_optional_declaration_code(self, entity_code, dll_linkage=None, template_params=None):
    return '__Pyx_Optional_Type<%s> %s' % (self.declaration_code('', False, dll_linkage, False, template_params), entity_code)