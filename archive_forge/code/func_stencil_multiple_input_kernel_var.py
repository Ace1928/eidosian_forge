import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
@stencil
def stencil_multiple_input_kernel_var(a, b, w):
    return w * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0] + b[0, 1] + b[1, 0] + b[0, -1] + b[-1, 0])