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
def parse_tuple_and_keywords(self, args, kws, fmt, keywords, *objs):
    charptr = ir.PointerType(ir.IntType(8))
    charptrary = ir.PointerType(charptr)
    argtypes = [self.pyobj, self.pyobj, charptr, charptrary]
    fnty = ir.FunctionType(ir.IntType(32), argtypes, var_arg=True)
    fn = self._get_function(fnty, name='PyArg_ParseTupleAndKeywords')
    return self.builder.call(fn, [args, kws, fmt, keywords] + list(objs))