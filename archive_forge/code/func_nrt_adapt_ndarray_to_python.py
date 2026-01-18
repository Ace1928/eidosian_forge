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
def nrt_adapt_ndarray_to_python(self, aryty, ary, dtypeptr):
    assert self.context.enable_nrt, 'NRT required'
    intty = ir.IntType(32)
    serial_aryty_pytype = self.unserialize(self.serialize_object(aryty.box_type))
    fnty = ir.FunctionType(self.pyobj, [self.voidptr, self.pyobj, intty, intty, self.pyobj])
    fn = self._get_function(fnty, name='NRT_adapt_ndarray_to_python_acqref')
    fn.args[0].add_attribute('nocapture')
    ndim = self.context.get_constant(types.int32, aryty.ndim)
    writable = self.context.get_constant(types.int32, int(aryty.mutable))
    aryptr = cgutils.alloca_once_value(self.builder, ary)
    return self.builder.call(fn, [self.builder.bitcast(aryptr, self.voidptr), serial_aryty_pytype, ndim, writable, dtypeptr])