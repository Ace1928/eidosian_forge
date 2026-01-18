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
def parse_tuple(self, args, fmt, *objs):
    charptr = ir.PointerType(ir.IntType(8))
    argtypes = [self.pyobj, charptr]
    fnty = ir.FunctionType(ir.IntType(32), argtypes, var_arg=True)
    fn = self._get_function(fnty, name='PyArg_ParseTuple')
    return self.builder.call(fn, [args, fmt] + list(objs))