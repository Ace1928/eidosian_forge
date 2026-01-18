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
def release_buffer(self, pbuf):
    fnty = ir.FunctionType(ir.VoidType(), [ir.PointerType(self.py_buffer_t)])
    fn = self._get_function(fnty, name='numba_release_buffer')
    return self.builder.call(fn, [pbuf])