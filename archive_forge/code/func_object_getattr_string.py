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
def object_getattr_string(self, obj, attr):
    cstr = self.context.insert_const_string(self.module, attr)
    fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.cstring])
    fn = self._get_function(fnty, name='PyObject_GetAttrString')
    return self.builder.call(fn, [obj, cstr])