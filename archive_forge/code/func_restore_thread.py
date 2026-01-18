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
def restore_thread(self, thread_state):
    """
        Restore the given thread state by reacquiring the GIL.
        """
    fnty = ir.FunctionType(ir.VoidType(), [self.voidptr])
    fn = self._get_function(fnty, name='PyEval_RestoreThread')
    self.builder.call(fn, [thread_state])