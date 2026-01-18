import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def in_html_mode():
    return isinstance(_Formatter, HTMLFormatter)