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
def sys_write_stdout(self, fmt, *args):
    fnty = ir.FunctionType(ir.VoidType(), [self.cstring], var_arg=True)
    fn = self._get_function(fnty, name='PySys_FormatStdout')
    return self.builder.call(fn, (fmt,) + args)