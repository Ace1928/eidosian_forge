import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def set_instance_variable(obj, varname, value, vartype):
    objc.object_setInstanceVariable.argtypes = [c_void_p, c_char_p, vartype]
    objc.object_setInstanceVariable(obj, ensure_bytes(varname), value)