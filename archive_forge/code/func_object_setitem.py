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
def object_setitem(self, obj, key, val):
    """
        obj[key] = val
        """
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj, self.pyobj])
    fn = self._get_function(fnty, name='PyObject_SetItem')
    return self.builder.call(fn, (obj, key, val))