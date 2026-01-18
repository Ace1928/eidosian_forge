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
def from_native_return(self, typ, val, env_manager):
    assert not isinstance(typ, types.Optional), 'callconv should have prevented the return of optional value'
    out = self.from_native_value(typ, val, env_manager)
    return out