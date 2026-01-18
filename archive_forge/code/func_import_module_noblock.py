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
def import_module_noblock(self, modname):
    fnty = ir.FunctionType(self.pyobj, [self.cstring])
    fn = self._get_function(fnty, name='PyImport_ImportModuleNoBlock')
    return self.builder.call(fn, [modname])