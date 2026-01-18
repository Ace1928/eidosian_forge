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
def object_get_private_data(self, obj):
    fnty = ir.FunctionType(self.voidptr, [self.pyobj])
    fn = self._get_function(fnty, name='numba_get_pyobject_private_data')
    return self.builder.call(fn, (obj,))