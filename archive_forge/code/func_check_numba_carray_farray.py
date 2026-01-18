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
def check_numba_carray_farray(self, usecase, dtype_usecase):
    pyfunc = usecase
    for sig in self.make_carray_sigs(carray_float32_usecase_sig):
        f = cfunc(sig)(pyfunc)
        self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)
    pyfunc = dtype_usecase
    for sig in self.make_carray_sigs(carray_float32_usecase_sig):
        f = cfunc(sig)(pyfunc)
        self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)
    with self.assertTypingError() as raises:
        f = cfunc(carray_float64_usecase_sig)(pyfunc)
    self.assertIn("mismatching dtype 'float32' for pointer type 'float64*'", str(raises.exception))
    pyfunc = dtype_usecase
    for sig in self.make_carray_sigs(carray_voidptr_usecase_sig):
        f = cfunc(sig)(pyfunc)
        self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)