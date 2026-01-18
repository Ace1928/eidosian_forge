import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def obj_to_string(a):
    out = io.StringIO()
    _PP(out, _Formatter(a))
    return out.getvalue()