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
def err_restore(self, ty, val, tb):
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj] * 3)
    fn = self._get_function(fnty, name='PyErr_Restore')
    return self.builder.call(fn, (ty, val, tb))