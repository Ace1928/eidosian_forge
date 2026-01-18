import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def insert_line_breaks(s, width):
    """Break s in lines of size width (approx)"""
    sz = len(s)
    if sz <= width:
        return s
    new_str = io.StringIO()
    w = 0
    for i in range(sz):
        if w > width and s[i] == ' ':
            new_str.write(u('<br />'))
            w = 0
        else:
            new_str.write(u(s[i]))
        w = w + 1
    return new_str.getvalue()