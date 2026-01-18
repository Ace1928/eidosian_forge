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
def float_as_double(self, fobj):
    fnty = ir.FunctionType(self.double, [self.pyobj])
    fn = self._get_function(fnty, name='PyFloat_AsDouble')
    return self.builder.call(fn, [fobj])