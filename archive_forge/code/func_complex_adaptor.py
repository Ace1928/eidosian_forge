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
def complex_adaptor(self, cobj, cmplx):
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, cmplx.type])
    fn = self._get_function(fnty, name='numba_complex_adaptor')
    return self.builder.call(fn, [cobj, cmplx])