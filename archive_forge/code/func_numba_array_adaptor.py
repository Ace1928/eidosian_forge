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
def numba_array_adaptor(self, ary, ptr):
    assert not self.context.enable_nrt
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.voidptr])
    fn = self._get_function(fnty, name='numba_adapt_ndarray')
    fn.args[0].add_attribute('nocapture')
    fn.args[1].add_attribute('nocapture')
    return self.builder.call(fn, (ary, ptr))