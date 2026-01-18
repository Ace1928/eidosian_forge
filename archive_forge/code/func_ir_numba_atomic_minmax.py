import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def ir_numba_atomic_minmax(T, Ti, NAN, OP, PTR_OR_VAL, FUNC):
    params = dict(T=T, Ti=Ti, NAN=NAN, OP=OP, PTR_OR_VAL=PTR_OR_VAL, FUNC=FUNC, CAS=ir_cas(Ti))
    return ir_numba_atomic_minmax_template.format(**params)