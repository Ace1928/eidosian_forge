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
def object_set_private_data(self, obj, ptr):
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj, self.voidptr])
    fn = self._get_function(fnty, name='numba_set_pyobject_private_data')
    return self.builder.call(fn, (obj, ptr))