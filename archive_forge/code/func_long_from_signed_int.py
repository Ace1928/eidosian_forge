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
def long_from_signed_int(self, ival):
    """
        Return a Python integer from any native integer value.
        """
    bits = ival.type.width
    if bits <= self.long.width:
        return self.long_from_long(self.builder.sext(ival, self.long))
    elif bits <= self.longlong.width:
        return self.long_from_longlong(self.builder.sext(ival, self.longlong))
    else:
        raise OverflowError('integer too big (%d bits)' % bits)