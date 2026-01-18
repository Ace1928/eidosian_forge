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
def get_regs_per_thread(self, signature=None):
    """
        Returns the number of registers used by each thread in this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get register
                          usage for. This may be omitted for a specialized
                          kernel.
        :return: The number of registers used by the compiled variant of the
                 kernel for the given signature and current device.
        """
    if signature is not None:
        return self.overloads[signature.args].regs_per_thread
    if self.specialized:
        return next(iter(self.overloads.values())).regs_per_thread
    else:
        return {sig: overload.regs_per_thread for sig, overload in self.overloads.items()}