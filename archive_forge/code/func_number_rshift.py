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
def number_rshift(self, lhs, rhs, inplace=False):
    return self._call_number_operator('Rshift', lhs, rhs, inplace=inplace)