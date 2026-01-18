import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
def make_cfarray_dtype_usecase(func):

    def cfarray_usecase(in_ptr, out_ptr, m, n):
        in_ = func(in_ptr, (m, n), dtype=np.float32)
        out = func(out_ptr, CARRAY_USECASE_OUT_LEN, np.float32)
        out[0] = in_.ndim
        out[1:3] = in_.shape
        out[3:5] = in_.strides
        out[5] = in_.flags.c_contiguous
        out[6] = in_.flags.f_contiguous
        s = 0
        for i, j in np.ndindex(m, n):
            s += in_[i, j] * (i - j)
        out[7] = s
    return cfarray_usecase