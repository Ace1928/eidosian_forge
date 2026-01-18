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
def list_setslice(self, lst, start, stop, obj):
    if obj is None:
        obj = self.get_null_object()
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.py_ssize_t, self.py_ssize_t, self.pyobj])
    fn = self._get_function(fnty, name='PyList_SetSlice')
    return self.builder.call(fn, (lst, start, stop, obj))