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
@contextlib.contextmanager
def if_object_ok(self, obj):
    with cgutils.if_likely(self.builder, cgutils.is_not_null(self.builder, obj)):
        yield