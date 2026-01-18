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
def nrt_adapt_buffer_from_python(self, buf, ptr):
    assert self.context.enable_nrt
    fnty = ir.FunctionType(ir.VoidType(), [ir.PointerType(self.py_buffer_t), self.voidptr])
    fn = self._get_function(fnty, name='NRT_adapt_buffer_from_python')
    fn.args[0].add_attribute('nocapture')
    fn.args[1].add_attribute('nocapture')
    return self.builder.call(fn, (buf, ptr))