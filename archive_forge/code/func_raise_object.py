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
def raise_object(self, exc=None):
    """
        Raise an arbitrary exception (type or value or (type, args)
        or None - if reraising).  A reference to the argument is consumed.
        """
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
    fn = self._get_function(fnty, name='numba_do_raise')
    if exc is None:
        exc = self.make_none()
    return self.builder.call(fn, (exc,))