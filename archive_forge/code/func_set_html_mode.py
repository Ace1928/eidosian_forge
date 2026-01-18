import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def set_html_mode(flag=True):
    global _Formatter
    if flag:
        _Formatter = HTMLFormatter()
    else:
        _Formatter = Formatter()