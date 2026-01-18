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
def string_as_string_and_size(self, strobj):
    """
        Returns a tuple of ``(ok, buffer, length)``.
        The ``ok`` is i1 value that is set if ok.
        The ``buffer`` is a i8* of the output buffer.
        The ``length`` is a i32/i64 (py_ssize_t) of the length of the buffer.
        """
    p_length = cgutils.alloca_once(self.builder, self.py_ssize_t)
    fnty = ir.FunctionType(self.cstring, [self.pyobj, self.py_ssize_t.as_pointer()])
    fname = 'PyUnicode_AsUTF8AndSize'
    fn = self._get_function(fnty, name=fname)
    buffer = self.builder.call(fn, [strobj, p_length])
    ok = self.builder.icmp_unsigned('!=', Constant(buffer.type, None), buffer)
    return (ok, buffer, self.builder.load(p_length))