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
def run_carray_usecase(self, pointer_factory, func):
    a = np.arange(10, 16).reshape((2, 3)).astype(np.float32)
    out = np.empty(CARRAY_USECASE_OUT_LEN, dtype=np.float32)
    func(pointer_factory(a), pointer_factory(out), *a.shape)
    return out