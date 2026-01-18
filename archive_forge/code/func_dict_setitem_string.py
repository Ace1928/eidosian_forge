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
def dict_setitem_string(self, dictobj, name, valobj):
    fnty = ir.FunctionType(ir.IntType(32), (self.pyobj, self.cstring, self.pyobj))
    fn = self._get_function(fnty, name='PyDict_SetItemString')
    cstr = self.context.insert_const_string(self.module, name)
    return self.builder.call(fn, (dictobj, cstr, valobj))