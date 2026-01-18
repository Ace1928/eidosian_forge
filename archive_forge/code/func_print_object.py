from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def print_object(self, obj):
    strobj = self.object_str(obj)
    cstr = self.string_as_string(strobj)
    fmt = self.context.insert_const_string(self.module, '%s')
    self.sys_write_stdout(fmt, cstr)
    self.decref(strobj)