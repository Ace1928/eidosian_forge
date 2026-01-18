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
def object_type(self, obj):
    """Emit a call to ``PyObject_Type(obj)`` to get the type of ``obj``.
        """
    fnty = ir.FunctionType(self.pyobj, [self.pyobj])
    fn = self._get_function(fnty, name='PyObject_Type')
    return self.builder.call(fn, (obj,))