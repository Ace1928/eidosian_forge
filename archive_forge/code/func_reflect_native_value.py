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
def reflect_native_value(self, typ, val, env_manager=None):
    """
        Reflect the native value onto its Python original, if any.
        An error bit (as an LLVM value) is returned.
        """
    impl = _reflectors.lookup(typ.__class__)
    if impl is None:
        return cgutils.false_bit
    is_error = cgutils.alloca_once_value(self.builder, cgutils.false_bit)
    c = _ReflectContext(self.context, self.builder, self, env_manager, is_error)
    impl(typ, val, c)
    return self.builder.load(c.is_error)