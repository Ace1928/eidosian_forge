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
def bool_from_long(self, ival):
    fnty = ir.FunctionType(self.pyobj, [self.long])
    fn = self._get_function(fnty, name='PyBool_FromLong')
    return self.builder.call(fn, [ival])