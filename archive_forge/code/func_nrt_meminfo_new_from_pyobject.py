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
def nrt_meminfo_new_from_pyobject(self, data, pyobj):
    """
        Allocate a new MemInfo with data payload borrowed from a python
        object.
        """
    mod = self.builder.module
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.voidptr_t, cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_meminfo_new_from_pyobject')
    fn.args[0].add_attribute('nocapture')
    fn.args[1].add_attribute('nocapture')
    fn.return_value.add_attribute('noalias')
    return self.builder.call(fn, [data, pyobj])