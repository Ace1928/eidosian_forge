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
def bytes_from_string_and_size(self, string, size):
    fnty = ir.FunctionType(self.pyobj, [self.cstring, self.py_ssize_t])
    fname = 'PyBytes_FromStringAndSize'
    fn = self._get_function(fnty, name=fname)
    return self.builder.call(fn, [string, size])