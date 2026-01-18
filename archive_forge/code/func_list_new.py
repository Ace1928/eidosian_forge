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
def list_new(self, szval):
    fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t])
    fn = self._get_function(fnty, name='PyList_New')
    return self.builder.call(fn, [szval])