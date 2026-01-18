from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def method_function_type(self):
    mflags = self.method_flags()
    kw = 'WithKeywords' if method_keywords in mflags else ''
    for m in mflags:
        if m == method_noargs or m == method_onearg:
            return 'PyCFunction'
        if m == method_varargs:
            return 'PyCFunction' + kw
        if m == method_fastcall:
            return '__Pyx_PyCFunction_FastCall' + kw
    return None