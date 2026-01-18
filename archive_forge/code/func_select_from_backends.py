import os
import sys
import warnings
from threading import RLock as threadRLock
from ctypes import CFUNCTYPE, c_int, CDLL, POINTER, c_uint
import numpy as np
import llvmlite.binding as ll
from llvmlite import ir
from numba.np.numpy_support import as_dtype
from numba.core import types, cgutils, config, errors
from numba.core.typing import signature
from numba.np.ufunc.wrappers import _wrapper_info
from numba.np.ufunc import ufuncbuilder
from numba.extending import overload, intrinsic
def select_from_backends(backends):
    """
                Selects from presented backends and returns the first working
                """
    lib = None
    for backend in backends:
        lib = select_known_backend(backend)
        if lib is not None:
            break
    else:
        backend = ''
    return (lib, backend)