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
def complex_imag_as_double(self, cobj):
    fnty = ir.FunctionType(ir.DoubleType(), [self.pyobj])
    fn = self._get_function(fnty, name='PyComplex_ImagAsDouble')
    return self.builder.call(fn, [cobj])