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
def list_getitem(self, lst, idx):
    """
        Returns a borrowed reference.
        """
    fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.py_ssize_t])
    fn = self._get_function(fnty, name='PyList_GetItem')
    if isinstance(idx, int):
        idx = self.context.get_constant(types.intp, idx)
    return self.builder.call(fn, [lst, idx])