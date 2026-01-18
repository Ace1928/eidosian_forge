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
def object_dump(self, obj):
    """
        Dump a Python object on C stderr.  For debugging purposes.
        """
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
    fn = self._get_function(fnty, name='_PyObject_Dump')
    return self.builder.call(fn, (obj,))