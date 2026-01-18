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
def bool_from_bool(self, bval):
    """
        Get a Python bool from a LLVM boolean.
        """
    longval = self.builder.zext(bval, self.long)
    return self.bool_from_long(longval)