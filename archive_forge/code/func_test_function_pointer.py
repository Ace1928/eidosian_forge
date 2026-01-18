import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_function_pointer(self):
    pyfunc = mod.use_func_pointer
    cfunc = jit(nopython=True)(pyfunc)
    for fa, fb, x in [(mod.cffi_sin, mod.cffi_cos, 1.0), (mod.cffi_sin, mod.cffi_cos, -1.0), (mod.cffi_cos, mod.cffi_sin, 1.0), (mod.cffi_cos, mod.cffi_sin, -1.0), (mod.cffi_sin_ool, mod.cffi_cos_ool, 1.0), (mod.cffi_sin_ool, mod.cffi_cos_ool, -1.0), (mod.cffi_cos_ool, mod.cffi_sin_ool, 1.0), (mod.cffi_cos_ool, mod.cffi_sin_ool, -1.0), (mod.cffi_sin, mod.cffi_cos_ool, 1.0), (mod.cffi_sin, mod.cffi_cos_ool, -1.0), (mod.cffi_cos, mod.cffi_sin_ool, 1.0), (mod.cffi_cos, mod.cffi_sin_ool, -1.0)]:
        expected = pyfunc(fa, fb, x)
        got = cfunc(fa, fb, x)
        self.assertEqual(got, expected)
    self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)