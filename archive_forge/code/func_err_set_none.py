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
def err_set_none(self, exctype):
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
    fn = self._get_function(fnty, name='PyErr_SetNone')
    if isinstance(exctype, str):
        exctype = self.get_c_object(exctype)
    return self.builder.call(fn, (exctype,))