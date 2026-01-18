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
def empty_declaration_code(self, pyrex=False):
    if pyrex:
        return self.declaration_code('', pyrex=True)
    if self._empty_declaration is None:
        self._empty_declaration = self.declaration_code('')
    return self._empty_declaration