import numpy as np
import os
import sys
import ctypes
import functools
from numba.core import config, serialize, sigutils, types, typing, utils
from numba.core.caching import Cache, CacheImpl
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaPerformanceWarning
from numba.core.typing.typeof import Purpose, typeof
from numba.cuda.api import get_current_device
from numba.cuda.args import wrap_arg
from numba.cuda.compiler import compile_cuda, CUDACompiler
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (missing_launch_config_msg,
from numba.cuda import types as cuda_types
from numba import cuda
from numba import _dispatcher
from warnings import warn
def get_call_template(self, args, kws):
    """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows resolution of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
    if self._can_compile:
        self.compile_device(tuple(args))
    func_name = self.py_func.__name__
    name = 'CallTemplate({0})'.format(func_name)
    call_template = typing.make_concrete_template(name, key=func_name, signatures=self.nopython_signatures)
    pysig = utils.pysignature(self.py_func)
    return (call_template, pysig, args, kws)