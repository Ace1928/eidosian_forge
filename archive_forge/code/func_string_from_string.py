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
def string_from_string(self, string):
    fnty = ir.FunctionType(self.pyobj, [self.cstring])
    fname = 'PyUnicode_FromString'
    fn = self._get_function(fnty, name=fname)
    return self.builder.call(fn, [string])