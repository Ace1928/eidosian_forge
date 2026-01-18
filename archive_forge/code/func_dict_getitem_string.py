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
def dict_getitem_string(self, dic, name):
    """Lookup name inside dict

        Returns a borrowed reference
        """
    fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.cstring])
    fn = self._get_function(fnty, name='PyDict_GetItemString')
    cstr = self.context.insert_const_string(self.module, name)
    return self.builder.call(fn, [dic, cstr])