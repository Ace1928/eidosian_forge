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
def slice_as_ints(self, obj):
    """
        Read the members of a slice of integers.

        Returns a (ok, start, stop, step) tuple where ok is a boolean and
        the following members are pointer-sized ints.
        """
    pstart = cgutils.alloca_once(self.builder, self.py_ssize_t)
    pstop = cgutils.alloca_once(self.builder, self.py_ssize_t)
    pstep = cgutils.alloca_once(self.builder, self.py_ssize_t)
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj] + [self.py_ssize_t.as_pointer()] * 3)
    fn = self._get_function(fnty, name='numba_unpack_slice')
    res = self.builder.call(fn, (obj, pstart, pstop, pstep))
    start = self.builder.load(pstart)
    stop = self.builder.load(pstop)
    step = self.builder.load(pstep)
    return (cgutils.is_null(self.builder, res), start, stop, step)