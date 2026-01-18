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
def emit_environment_sentry(self, envptr, return_pyobject=False, debug_msg=''):
    """Emits LLVM code to ensure the `envptr` is not NULL
        """
    is_null = cgutils.is_null(self.builder, envptr)
    with cgutils.if_unlikely(self.builder, is_null):
        if return_pyobject:
            fnty = self.builder.function.type.pointee
            assert fnty.return_type == self.pyobj
            self.err_set_string('PyExc_RuntimeError', f'missing Environment: {debug_msg}')
            self.builder.ret(self.get_null_object())
        else:
            self.context.call_conv.return_user_exc(self.builder, RuntimeError, (f'missing Environment: {debug_msg}',))