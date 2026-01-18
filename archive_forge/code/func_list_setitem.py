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
def list_setitem(self, lst, idx, val):
    """
        Warning: Steals reference to ``val``
        """
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.py_ssize_t, self.pyobj])
    fn = self._get_function(fnty, name='PyList_SetItem')
    return self.builder.call(fn, [lst, idx, val])