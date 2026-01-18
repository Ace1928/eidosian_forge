import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_const(self, a):
    k = a.decl().kind()
    if k == Z3_OP_RE_EMPTY_SET:
        return self.pp_set('Empty', a)
    elif k == Z3_OP_SEQ_EMPTY:
        return self.pp_set('Empty', a)
    elif k == Z3_OP_RE_FULL_SET:
        return self.pp_set('Full', a)
    elif k == Z3_OP_CHAR_CONST:
        return self.pp_char(a)
    return self.pp_name(a)