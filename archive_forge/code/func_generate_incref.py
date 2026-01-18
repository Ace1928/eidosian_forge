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
def generate_incref(self, code, cname, nanny):
    if nanny:
        code.funcstate.needs_refnanny = True
        code.putln('__Pyx_INCREF(%s);' % self.as_pyobject(cname))
    else:
        code.putln('Py_INCREF(%s);' % self.as_pyobject(cname))