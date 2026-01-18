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
def gil_release(self, gil):
    """
        Release the acquired GIL by gil_ensure().
        Must be paired with a gil_ensure().
        """
    gilptrty = ir.PointerType(self.gil_state)
    fnty = ir.FunctionType(ir.VoidType(), [gilptrty])
    fn = self._get_function(fnty, 'numba_gil_release')
    return self.builder.call(fn, [gil])