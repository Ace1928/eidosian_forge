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
def save_thread(self):
    """
        Release the GIL and return the former thread state
        (an opaque non-NULL pointer).
        """
    fnty = ir.FunctionType(self.voidptr, [])
    fn = self._get_function(fnty, name='PyEval_SaveThread')
    return self.builder.call(fn, [])