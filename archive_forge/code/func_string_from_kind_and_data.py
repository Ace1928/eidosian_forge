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
def string_from_kind_and_data(self, kind, string, size):
    fnty = ir.FunctionType(self.pyobj, [ir.IntType(32), self.cstring, self.py_ssize_t])
    fname = 'PyUnicode_FromKindAndData'
    fn = self._get_function(fnty, name=fname)
    return self.builder.call(fn, [kind, string, size])