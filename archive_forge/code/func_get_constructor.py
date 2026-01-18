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
def get_constructor(self, pos):
    constructor = self.scope.lookup('<init>')
    if constructor is not None:
        return constructor
    nogil = True
    for base in self.base_classes:
        base_constructor = base.scope.lookup('<init>')
        if base_constructor and (not base_constructor.type.nogil):
            nogil = False
            break
    func_type = CFuncType(self, [], exception_check='+', nogil=nogil)
    return self.scope.declare_cfunction(u'<init>', func_type, pos)