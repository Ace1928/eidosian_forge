import sys
import errno
import functools
import atexit
from multiprocessing.util import (  # noqa
from .compat import get_errno
def get_pdeathsig():
    """
    Return the current value of the parent process death signal
    """
    if not sys.platform.startswith('linux'):
        raise OSError()
    try:
        if 'cffi' in sys.modules:
            ffi = cffi.FFI()
            ffi.cdef('int prctl (int __option, ...);')
            arg = ffi.new('int *')
            C = ffi.dlopen(None)
            C.prctl(PR_GET_PDEATHSIG, arg)
            return arg[0]
        else:
            sig = ctypes.c_int()
            libc = ctypes.cdll.LoadLibrary('libc.so.6')
            libc.prctl(PR_GET_PDEATHSIG, ctypes.byref(sig))
            return sig.value
    except Exception:
        raise OSError()