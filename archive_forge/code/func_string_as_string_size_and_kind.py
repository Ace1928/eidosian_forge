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
def string_as_string_size_and_kind(self, strobj):
    """
        Returns a tuple of ``(ok, buffer, length, kind)``.
        The ``ok`` is i1 value that is set if ok.
        The ``buffer`` is a i8* of the output buffer.
        The ``length`` is a i32/i64 (py_ssize_t) of the length of the buffer.
        The ``kind`` is a i32 (int32) of the Unicode kind constant
        The ``hash`` is a long/uint64_t (py_hash_t) of the Unicode constant hash
        """
    p_length = cgutils.alloca_once(self.builder, self.py_ssize_t)
    p_kind = cgutils.alloca_once(self.builder, ir.IntType(32))
    p_ascii = cgutils.alloca_once(self.builder, ir.IntType(32))
    p_hash = cgutils.alloca_once(self.builder, self.py_hash_t)
    fnty = ir.FunctionType(self.cstring, [self.pyobj, self.py_ssize_t.as_pointer(), ir.IntType(32).as_pointer(), ir.IntType(32).as_pointer(), self.py_hash_t.as_pointer()])
    fname = 'numba_extract_unicode'
    fn = self._get_function(fnty, name=fname)
    buffer = self.builder.call(fn, [strobj, p_length, p_kind, p_ascii, p_hash])
    ok = self.builder.icmp_unsigned('!=', Constant(buffer.type, None), buffer)
    return (ok, buffer, self.builder.load(p_length), self.builder.load(p_kind), self.builder.load(p_ascii), self.builder.load(p_hash))