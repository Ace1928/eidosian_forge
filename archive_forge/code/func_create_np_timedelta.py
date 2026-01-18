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
def create_np_timedelta(self, val, unit_code):
    unit_code = Constant(ir.IntType(32), int(unit_code))
    fnty = ir.FunctionType(self.pyobj, [ir.IntType(64), ir.IntType(32)])
    fn = self._get_function(fnty, name='numba_create_np_timedelta')
    return self.builder.call(fn, [val, unit_code])