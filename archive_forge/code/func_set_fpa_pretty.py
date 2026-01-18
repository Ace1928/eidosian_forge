import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def set_fpa_pretty(flag=True):
    global _Formatter
    global _z3_op_to_str
    _Formatter.fpa_pretty = flag
    if flag:
        for _k, _v in _z3_op_to_fpa_pretty_str.items():
            _z3_op_to_str[_k] = _v
        for _k in _z3_fpa_infix:
            _infix_map[_k] = True
    else:
        for _k, _v in _z3_op_to_fpa_normal_str.items():
            _z3_op_to_str[_k] = _v
        for _k in _z3_fpa_infix:
            _infix_map[_k] = False