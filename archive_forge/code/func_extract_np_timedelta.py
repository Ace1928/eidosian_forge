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
def extract_np_timedelta(self, obj):
    fnty = ir.FunctionType(ir.IntType(64), [self.pyobj])
    fn = self._get_function(fnty, name='numba_extract_np_timedelta')
    return self.builder.call(fn, [obj])