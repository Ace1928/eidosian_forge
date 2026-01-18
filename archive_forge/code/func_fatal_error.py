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
def fatal_error(self, msg):
    fnty = ir.FunctionType(ir.VoidType(), [self.cstring])
    fn = self._get_function(fnty, name='Py_FatalError')
    fn.attributes.add('noreturn')
    cstr = self.context.insert_const_string(self.module, msg)
    self.builder.call(fn, (cstr,))