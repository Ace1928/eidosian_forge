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
def set_new(self, iterable=None):
    if iterable is None:
        iterable = self.get_null_object()
    fnty = ir.FunctionType(self.pyobj, [self.pyobj])
    fn = self._get_function(fnty, name='PySet_New')
    return self.builder.call(fn, [iterable])