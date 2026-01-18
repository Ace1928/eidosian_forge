import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def check_1d_slicing_with_arg(self, pyfunc, flags):
    args = list(range(-9, 10))
    arraytype = types.Array(types.int32, 1, 'C')
    argtys = (arraytype, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(10, dtype='i4')
    for arg in args:
        self.assertEqual(pyfunc(a, arg), cfunc(a, arg))
    arraytype = types.Array(types.int32, 1, 'A')
    argtys = (arraytype, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(20, dtype='i4')[::2]
    self.assertFalse(a.flags['C_CONTIGUOUS'])
    self.assertFalse(a.flags['F_CONTIGUOUS'])
    for arg in args:
        self.assertEqual(pyfunc(a, arg), cfunc(a, arg))