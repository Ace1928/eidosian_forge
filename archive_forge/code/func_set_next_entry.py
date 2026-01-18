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
def set_next_entry(self, set, posptr, keyptr, hashptr):
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.py_ssize_t.as_pointer(), self.pyobj.as_pointer(), self.py_hash_t.as_pointer()])
    fn = self._get_function(fnty, name='_PySet_NextEntry')
    return self.builder.call(fn, (set, posptr, keyptr, hashptr))