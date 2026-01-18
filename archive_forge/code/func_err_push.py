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
def err_push(self, keep_new=False):
    """
        Temporarily push the current error indicator while the code
        block is executed.  If *keep_new* is True and the code block
        raises a new error, the new error is kept, otherwise the old
        error indicator is restored at the end of the block.
        """
    pty, pval, ptb = [cgutils.alloca_once(self.builder, self.pyobj) for i in range(3)]
    self.err_fetch(pty, pval, ptb)
    yield
    ty = self.builder.load(pty)
    val = self.builder.load(pval)
    tb = self.builder.load(ptb)
    if keep_new:
        new_error = cgutils.is_not_null(self.builder, self.err_occurred())
        with self.builder.if_else(new_error, likely=False) as (if_error, if_ok):
            with if_error:
                self.decref(ty)
                self.decref(val)
                self.decref(tb)
            with if_ok:
                self.err_restore(ty, val, tb)
    else:
        self.err_restore(ty, val, tb)