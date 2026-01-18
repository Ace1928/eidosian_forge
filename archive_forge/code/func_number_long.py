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
def number_long(self, numobj):
    fnty = ir.FunctionType(self.pyobj, [self.pyobj])
    fn = self._get_function(fnty, name='PyNumber_Long')
    return self.builder.call(fn, [numobj])