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
def err_set_string(self, exctype, msg):
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj, self.cstring])
    fn = self._get_function(fnty, name='PyErr_SetString')
    if isinstance(exctype, str):
        exctype = self.get_c_object(exctype)
    if isinstance(msg, str):
        msg = self.context.insert_const_string(self.module, msg)
    return self.builder.call(fn, (exctype, msg))